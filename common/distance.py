from pykeops.torch import LazyTensor


class L2_DIS:
    factor = 1 / 32
    
    @staticmethod
    def __call__(X, Y, use_keops=False):
        """
        X.shape = (batch, M, D)
        Y.shape = (batch, N, D)
        returned cost matrix's shape is (batch, M, N)
        """
        X, Y = X.float(), Y.float()
        if use_keops:
            X, Y = X.contiguous(), Y.contiguous()
            x_i = LazyTensor(X[:, :, None, :])  # (B,N,1,D)
            y_j = LazyTensor(Y[:, None, :, :])  # (B,1,M,D)
        else:
            x_i = X.unsqueeze(-2)
            y_j = Y.unsqueeze(-3)
        
        C = ((x_i - y_j) ** 2).sum(dim=-1) / 2
        
        return C * L2_DIS.factor
    
    @staticmethod
    def barycenter(weight, coord):
        """
        weight.shape = (batch, M, N)
        coord.shape = (batch, M, D)
        returned coord's shape is (batch, N, D)
        """
        weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-12)
        
        return weight.permute(0, 2, 1) @ coord
