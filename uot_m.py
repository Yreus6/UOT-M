import torch
import numpy as np
from pykeops.torch import LazyTensor

from common.distance import L2_DIS
from common.utils import normalize_denmap, den2coord, init_dot


def max_diameter(x, y):
    D = x.shape[-1]
    x, y = x.view(-1, D), y.view(-1, D)
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    
    return diameter


def epsilon_schedule(diameter, blur, scaling):
    schedule = np.arange(np.log(diameter), np.log(blur), np.log(scaling))
    epsilon_s = [diameter] + [np.exp(e) for e in schedule] + [blur]
    
    return epsilon_s


def dampening(eps, rho):
    return 1 if rho is None else 1 / (1 + eps / rho)


def log_weights(a):
    a_log = a.log()
    a_log[a <= 0] = -100000
    
    return a_log


def scaling_parameters(x, y, blur, reach, diameter, scaling):
    if diameter is None:
        diameter = max_diameter(x, y)
    
    eps = blur
    rho = None if reach is None else reach
    eps_list = epsilon_schedule(diameter, blur, scaling)
    
    return diameter, eps, eps_list, rho


def softmin(logB, C, epsilon, G=None):
    B = C.shape[0]
    
    if G is None:
        h_y = logB.view(B, -1)
    else:
        h_y = logB.view(B, -1) + G.view(B, -1) / epsilon
    h_y = h_y.contiguous()
    h_j = LazyTensor(h_y[:, None, :, None])  # (B, 1, M, 1)
    
    s = h_j - C * torch.tensor([1 / epsilon]).type_as(h_y)
    smin = s.logsumexp(2).view(B, -1, 1)
    
    return -epsilon * smin


class UnbalancedWeight(torch.nn.Module):
    def __init__(self, eps, rho):
        super(UnbalancedWeight, self).__init__()
        self.eps, self.rho = eps, rho
    
    def forward(self, x):
        return (self.rho + self.eps / 2) * x
    
    def backward(self, g):
        return (self.rho + self.eps) * g


def scal(a, f, batch=False):
    if batch:
        B = a.shape[0]
        return (a.reshape(B, -1) * f.reshape(B, -1)).sum(1)
    else:
        return torch.dot(a.reshape(-1), f.reshape(-1))


class ScaleOT:
    def __init__(self, blur=0.01, scaling=0.5, reach=None, debias=False):
        self.blur = blur
        self.scaling = scaling
        self.reach = reach
        self.debias = debias
    
    @torch.no_grad()
    def __call__(self, A, B, X, Y, cost, diameter=None):
        C = cost(X, Y, False)
        diameter = diameter if diameter is not None else C.max().item()
        diameter = max(8, diameter)
        diameter, eps, eps_list, rho = scaling_parameters(X, Y, self.blur, self.reach, diameter, self.scaling)
        logA, logB = log_weights(A), log_weights(B)
        
        C_xy, C_yx = cost(X, Y, True), cost(Y, X, True)
        if self.debias:
            C_xx, C_yy = cost(X, X, True), cost(Y, Y, True)
        
        damping = dampening(diameter, rho)
        
        F_ba = damping * softmin(logB, C_xy, diameter)
        G_ab = damping * softmin(logA, C_yx, diameter)
        if self.debias:
            F_aa = damping * softmin(logA, C_xx, diameter)
            G_bb = damping * softmin(logB, C_yy, diameter)
        
        for i, epsilon in enumerate(eps_list):
            damping = dampening(epsilon, rho)
            Ft_ba = damping * softmin(logB, C_xy, epsilon, G_ab)
            Gt_ab = damping * softmin(logA, C_yx, epsilon, F_ba)
            if self.debias:
                Ft_aa = damping * softmin(logA, C_xx, epsilon, F_aa)
                Gt_bb = damping * softmin(logB, C_yy, epsilon, G_bb)
            
            F_ba, G_ab = 0.5 * (F_ba + Ft_ba), 0.5 * (G_ab + Gt_ab)
            if self.debias:
                F_aa, G_bb = 0.5 * (F_aa + Ft_aa), 0.5 * (G_bb + Gt_bb)
        
        damping = dampening(eps, rho)
        F_ba, G_ab = (
            damping * softmin(logB, C_xy, eps, G_ab),
            damping * softmin(logA, C_yx, eps, F_ba),
        )
        
        if self.debias:
            F_aa = damping * softmin(logA, C_xx, eps, F_aa)
            G_bb = damping * softmin(logB, C_yy, eps, G_bb)
            l = self.loss(A, B, eps, rho, F_ba, G_ab, F_aa, G_bb)
            
            return l, F_ba - F_aa, G_ab - G_bb
        
        l = self.loss(A, B, eps, rho, F_ba, G_ab)
        
        return l, F_ba, G_ab
    
    def plan(self, A, B, X, Y, F, G, cost):
        C = cost(X, Y)
        PI1 = torch.exp((F + G.permute(0, 2, 1) - C).detach() / self.blur)
        PI2 = A * B.permute(0, 2, 1)
        PI = PI1 * PI2
        
        return PI
    
    def loss(self, A, B, eps, rho, F_ba, G_ab, F_aa=None, G_bb=None):
        if self.reach is None:
            if self.debias:
                return scal(A, F_ba - F_aa, True) + scal(B, G_ab - G_bb, True)
            
            return scal(A, F_ba, batch=True) + scal(B, G_ab, batch=True)
        else:
            if self.debias:
                return scal(
                    A,
                    UnbalancedWeight(eps, rho)(
                        (-F_aa / rho).exp() - (-F_ba / rho).exp()
                    ),
                    batch=True,
                ) + scal(
                    B,
                    UnbalancedWeight(eps, rho)(
                        (-G_bb / rho).exp() - (-G_ab / rho).exp()
                    ),
                    batch=True,
                )
            
            return scal(
                A, UnbalancedWeight(eps, rho)(1 - (-F_ba / rho).exp()), batch=True
            ) + scal(
                B, UnbalancedWeight(eps, rho)(1 - (-G_ab / rho).exp()), batch=True
            )


blur = 0.01
per_cost = L2_DIS()
reach = 15000
tau = 0.05
ot = ScaleOT(blur=blur, scaling=0.9, reach=reach, debias=False)


@torch.no_grad()
def UOT_M(A, A_coord, B, B_coord, max_itern=8):
    best_l = 1e12
    
    for iter in range(max_itern):
        # OT-step
        l, F, G = ot(A, B, A_coord, B_coord, per_cost)
        PI = ot.plan(A, B, A_coord, B_coord, F, G, per_cost)
        
        entropy = torch.mean((1e-20 + PI) * torch.log(1e-20 + PI))
        point_loss = torch.sum((PI.sum(1).reshape(1, -1, 1) - B) ** 2)
        pixel_loss = torch.sum((PI.sum(2).reshape(1, -1, 1) - A) ** 2)
        l += blur * entropy + tau * (point_loss + pixel_loss)
        
        PI = PI[0]
        n, m = PI.shape
        new_m = int(PI.sum().item() + 0.5)
        
        if new_m > m:
            padding = (0, new_m - m)
            nPI = torch.nn.functional.pad(PI, padding, 'constant', 0)
        elif new_m < m:
            cropped_PI = PI[:, :new_m]
            
            original_sum = PI.sum()
            cropped_sum = cropped_PI.sum()
            scaling_factor = original_sum / cropped_sum
            
            nPI = cropped_PI * scaling_factor
        else:
            nPI = PI
        
        nPI = nPI.unsqueeze(0)
        
        # M-step
        nB_coord = per_cost.barycenter(nPI, A_coord)
        
        B_coord = nB_coord
        B = torch.ones(1, new_m, 1).to(B_coord)
        
        if l.item() < best_l:
            best_l = l.item()
            best_B_coord = B_coord
    
    return best_B_coord.reshape(-1, 2)


@torch.no_grad()
def den2seq(denmap, scale_factor=8, max_itern=16, ot_scaling=0.75):
    ot.scaling = ot_scaling
    num, norm_den = normalize_denmap(denmap)
    if norm_den is None:
        return torch.zeros((0, 2)).to(denmap)
    
    A, A_coord = den2coord(norm_den, scale_factor)
    B, B_coord = init_dot(norm_den, num, scale_factor)
    flocs = UOT_M(A, A_coord, B, B_coord, max_itern=max_itern)
    
    return flocs
