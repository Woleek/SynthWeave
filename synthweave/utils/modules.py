from torch import nn
import torch.nn.functional as F

class LazyLinearXavier(nn.LazyLinear):
    def __init__(self, out_features: int, bias: bool = True):
        super(LazyLinearXavier, self).__init__(out_features, bias)

    def reset_parameters(self):
        if not self.has_uninitialized_params() and self.in_features != 0:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.01)
                
class LinearXavier(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LinearXavier, self).__init__(in_features, out_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.01)
            
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True, dropout: float = 0.5):
        super(ProjectionMLP, self).__init__()
        
        self.proj = nn.Sequential(
            LinearXavier(in_dim, hidden_dim, bias),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            LinearXavier(hidden_dim, out_dim, bias)
        )
        
    def forward(self, x):
        return self.proj(x)
    
class LazyProjectionMLP(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, bias: bool = True, dropout: float = 0.5):
        super(LazyProjectionMLP, self).__init__()
        
        self.proj = nn.Sequential(
            LazyLinearXavier(hidden_dim, bias),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            LinearXavier(hidden_dim, out_dim, bias)
        )
        
    def forward(self, x):
        return self.proj(x)
    
class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=-1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)