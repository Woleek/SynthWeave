from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict

class GaussianNoise(nn.Module):
    def __init__(self, noise_std: float = 0.0):
        """
        Args:
            noise_std: Standard deviation of the Gaussian noise to add.
                       If 0.0, this module is effectively a no-op.
        """
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0.0:
            return x + torch.randn_like(x) * self.noise_std
        return x

class VIB(nn.Module):
    """
    Variational Information Bottleneck layer.
    Given an input tensor of dimension `in_dim`, projects to a bottleneck of `bottleneck_dim`
    by learning μ and log(σ²), sampling z = μ + σ * ε, and returning the KL divergence loss.
    """
    def __init__(self, in_dim: int, bottleneck_dim: int):
        super().__init__()
        self.mu_layer = nn.Linear(in_dim, bottleneck_dim)
        self.logvar_layer = nn.Linear(in_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            z: Tensor of shape (batch_size, bottleneck_dim)
            kl_loss: scalar tensor (mean KL divergence over batch)
        """
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Compute KL divergence between N(mu, sigma^2) and N(0, 1)
        kl_element = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())   # shape (batch, D)
        # Mean over latent dims, then mean over batch
        kl_loss    = kl_element.mean()
        return z, kl_loss
    
class CholeskyWhitening(nn.Module):
    """
    Feature whitening using Cholesky decomposition with numerical stability.
    
    This module computes the whitening transformation that decorrelates features
    and normalizes their variance to 1. Uses Cholesky decomposition for efficiency
    and includes regularization for numerical stability.
    
    Args:
        num_features: Number of input features
        eps: Regularization parameter for numerical stability (default: 1e-5)
        momentum: Momentum for running statistics (default: 0.1)
        track_running_stats: Whether to track running mean/covariance (default: True)
        affine: Whether to include learnable affine parameters (default: True)
        mode: Whitening mode - 'ZCA' (zero-phase component analysis) or 'PCA' (default: 'ZCA')
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        affine: bool = True,
        mode: str = 'ZCA'
    ):
        super(CholeskyWhitening, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.affine = affine
        self.mode = mode.upper()
        
        if self.mode not in ['ZCA', 'PCA']:
            raise ValueError("Mode must be either 'ZCA' or 'PCA'")
        
        # Learnable affine parameters
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Running statistics
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_cov', torch.eye(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_cov', None)
            self.register_buffer('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset parameters and running statistics."""
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_cov.copy_(torch.eye(self.num_features))
            self.num_batches_tracked.zero_()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    
    def _compute_whitening_matrix(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Compute whitening matrix using Cholesky decomposition.
        
        Args:
            cov: Covariance matrix [num_features, num_features]
            
        Returns:
            Whitening transformation matrix
        """
        # Add regularization for numerical stability
        reg_cov = cov + self.eps * torch.eye(
            self.num_features, 
            device=cov.device, 
            dtype=cov.dtype
        )
        
        try:
            # Cholesky decomposition: reg_cov = L @ L.T
            L = torch.linalg.cholesky(reg_cov)
            
            # Whitening matrix is L^{-1}
            whitening_matrix = torch.linalg.solve_triangular(
                torch.eye(self.num_features, device=L.device, dtype=L.dtype), 
                L, 
                upper=False
            )
            
            if self.mode == 'ZCA':
                # For ZCA whitening, we need the symmetric whitening matrix
                # ZCA = U @ diag(1/sqrt(eigenvals)) @ U.T
                # But we can compute it via: ZCA = W.T @ W where W is PCA whitening
                # However, with Cholesky we get: whitening_matrix = L^{-1}
                # For ZCA, we need: cov^{-1/2} which is symmetric
                
                # Use eigendecomposition for ZCA mode
                eigenvals, eigenvecs = torch.linalg.eigh(reg_cov)
                eigenvals = torch.clamp(eigenvals, min=self.eps)
                
                whitening_matrix = eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
            
            return whitening_matrix
            
        except RuntimeError as e:
            # Fallback to eigendecomposition if Cholesky fails
            print(f"Cholesky decomposition failed: {e}. Falling back to eigendecomposition.")
            eigenvals, eigenvecs = torch.linalg.eigh(reg_cov)
            eigenvals = torch.clamp(eigenvals, min=self.eps)
            
            if self.mode == 'ZCA':
                whitening_matrix = eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
            else:  # PCA mode
                whitening_matrix = torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
            
            return whitening_matrix
    
    def _update_running_stats(self, mean: torch.Tensor, cov: torch.Tensor) -> None:
        """Update running statistics with exponential moving average."""
        if self.num_batches_tracked == 0:
            self.running_mean.copy_(mean)
            self.running_cov.copy_(cov)
        else:
            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_cov.mul_(1 - self.momentum).add_(cov, alpha=self.momentum)
        
        self.num_batches_tracked += 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of whitening transformation.
        
        Args:
            x: Input tensor of shape [batch_size, num_features] or 
               [batch_size, num_features, height, width] for 2D whitening
               
        Returns:
            Whitened features of the same shape as input
        """
        # Handle different input shapes
        original_shape = x.shape
        if x.dim() == 4:  # [B, C, H, W]
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [B, H*W, C]
            x = x.contiguous().view(-1, channels)  # [B*H*W, C]
        elif x.dim() == 2:  # [B, C]
            pass
        else:
            raise ValueError(f"Input must be 2D or 4D tensor, got {x.dim()}D")
        
        batch_size = x.shape[0]
        
        if self.training or not self.track_running_stats:
            # Compute batch statistics
            mean = x.mean(dim=0)  # [num_features]
            x_centered = x - mean
            
            # Compute covariance matrix
            cov = torch.mm(x_centered.T, x_centered) / (batch_size - 1)
            
            # Update running statistics
            if self.track_running_stats and self.training:
                with torch.no_grad():
                    self._update_running_stats(mean, cov)
        
        else:
            # Use running statistics
            mean = self.running_mean
            cov = self.running_cov
            x_centered = x - mean
        
        # Compute whitening matrix
        with torch.amp.autocast("cuda", dtype=torch.float32):
            whitening_matrix = self._compute_whitening_matrix(cov)
        
        # Apply whitening transformation
        x_whitened = torch.mm(x_centered, whitening_matrix.T)
        
        # Apply affine transformation if enabled
        if self.affine:
            x_whitened = x_whitened * self.weight + self.bias
        
        # Reshape back to original shape
        if len(original_shape) == 4:
            x_whitened = x_whitened.view(original_shape[0], height * width, channels)
            x_whitened = x_whitened.permute(0, 2, 1).view(original_shape)
        
        return x_whitened
    
    def extra_repr(self) -> str:
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}, '
                f'mode={self.mode}')

class LazyLinearXavier(nn.LazyLinear):
    """A lazy linear layer with Xavier uniform initialization.

    This layer automatically determines its input size on first use and
    initializes weights using Xavier uniform initialization.
    """

    def __init__(self, out_features: int, bias: bool = True):
        """
        Args:
            out_features (int): Size of output features
            bias (bool, optional): If True, adds a learnable bias. Defaults to True.
        """
        super(LazyLinearXavier, self).__init__(out_features, bias)

    def reset_parameters(self):
        """Initializes the layer parameters using Xavier uniform initialization."""
        if not self.has_uninitialized_params() and self.in_features != 0:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.01)


class LinearXavier(nn.Linear):
    """A linear layer with Xavier uniform initialization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
            bias (bool, optional): If True, adds a learnable bias. Defaults to True.
        """
        super(LinearXavier, self).__init__(in_features, out_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the layer parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.01)


class ProjectionMLP(nn.Module):
    """A projection MLP with two linear layers, batch normalization, ReLU, and dropout.

    The architecture follows: Linear -> BatchNorm -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension
            bias (bool, optional): If True, adds learnable bias to linear layers. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
        """
        super(ProjectionMLP, self).__init__()

        self.proj = nn.Sequential(
            OrderedDict([
                    ("linear1", nn.Linear(in_dim, in_dim)),
                    ("bn1",   nn.BatchNorm1d(in_dim)),
                    ("relu1",   nn.LeakyReLU()),
                    ("drop1",   nn.Dropout(dropout)),

                    ("linear2", nn.Linear(in_dim, out_dim)),
                    # ("relu2",   nn.LeakyReLU()),
                    # ("drop2",   nn.Dropout(dropout)),

                    # ("linear3", nn.Linear(out_dim, out_dim)),
                ]
            )
        )

    def forward(self, x):
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.proj(x)


class LazyProjectionMLP(nn.Module):
    """A lazy projection MLP with two linear layers, batch normalization, ReLU, and dropout.

    Similar to ProjectionMLP but with a lazy first layer that infers input dimension.
    """

    def __init__(
        self, hidden_dim: int, out_dim: int, bias: bool = True, dropout: float = 0.5
    ):
        """
        Args:
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension
            bias (bool, optional): If True, adds learnable bias to linear layers. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
        """
        super(LazyProjectionMLP, self).__init__()

        self.proj = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.LazyLinear(hidden_dim, bias)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("linear2", nn.Linear(hidden_dim, out_dim, bias)),
                ]
            )
        )

    def forward(self, x):
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.proj(x)


class L2NormalizationLayer(nn.Module):
    """A layer that performs L2 normalization on the input tensor."""

    def __init__(self, dim=1, eps=1e-12):
        """
        Args:
            dim (int, optional): Dimension along which to normalize. Defaults to -1.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-12.
        """
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        """Forward pass performing L2 normalization.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: L2 normalized tensor
        """
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
