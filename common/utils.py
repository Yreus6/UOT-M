import random

import torch
import torch.nn.functional as F


def normalize_denmap(denmap):
    assert denmap.dim() == 2, f'the shape of density map should be [H, W], but the given one is {denmap.shape}'
    
    num = int(denmap.sum().item() + 0.5)
    if num < 0.5:
        return num, None
    
    # normalize density map
    denmap = denmap * num / denmap.sum()
    
    return num, denmap


def den2coord(denmap, scale_factor=8):
    coord = torch.nonzero(denmap > 1e-12)
    denval = denmap[coord[:, 0], coord[:, 1]]
    if scale_factor != 1:
        coord = coord.float() * scale_factor + scale_factor / 2
    
    return denval.reshape(1, -1, 1), coord.reshape(1, -1, 2)


def init_dot(denmap, n, scale_factor=8):
    norm_den = denmap[None, None, ...]
    norm_den = F.interpolate(norm_den, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    norm_den = norm_den[0, 0]
    
    d_coord = torch.nonzero(norm_den > 1e-12)
    norm_den = norm_den[d_coord[:, 0], d_coord[:, 1]]
    
    cidx = torch.tensor(random.sample(range(norm_den.shape[-1]), n))
    coord = d_coord[cidx]
    
    B = torch.ones(1, n, 1).to(denmap)
    B_coord = coord.reshape(1, n, 2)
    
    return B, B_coord
