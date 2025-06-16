from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ------------------------------------------------------------------
# 1. Identity Attenuation (FRIDAY-style)
# ------------------------------------------------------------------

class IdentityAttenuationLoss(nn.Module):
    """
    |L_IA| = |cos(z_f, z_id)|   (mean over the mini-batch)
    """
    def forward(self, z_f: torch.Tensor, z_id: torch.Tensor) -> torch.Tensor:  # z_f, z_id: (B, d)
        z_f = F.normalize(z_f, dim=-1)
        z_id = F.normalize(z_id, dim=-1)
        cosine = (z_f * z_id).sum(dim=-1)             # (B,)
        return cosine.abs().mean()

# Usage ------------------------------------------------------------
# id_net is your *pre-trained & frozen* face-recognition branch
# z_id = id_net(inputs).detach()
# loss_ia = λ_IA * IdentityAttenuationLoss()(z_f, z_id)
# ------------------------------------------------------------------
# 2. Gradient Reversal + Decomposition (CrossDF-style)
# ------------------------------------------------------------------

class _GradRevFn(Function):
    @staticmethod
    def forward(ctx, x, beta):        # beta > 0
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.beta * g, None    # flip & scale the gradient

class GradientReversal(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    def forward(self, x):
        return _GradRevFn.apply(x, self.beta)

class Decomposer(nn.Module):
    """
    Learns a mask M_df ∈ (0,1)^d, then splits z_f into:
        z_df = M_df ⊙ z_f
        z_os = (1-M_df) ⊙ z_f
    """
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.mask_net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid()
        )

    def forward(self, z_f: torch.Tensor):
        mask = self.mask_net(z_f)             # (B, d)
        z_df = mask * z_f
        z_os = (1 - mask) * z_f
        return z_df, z_os, mask               # mask is optional for viz/debug

class MIDiscriminator(nn.Module):
    """
    T_φ that scores joint vs. shuffled pairs.
    """
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)                    # logit
        )
    def forward(self, z_df, z_os):
        return self.net(torch.cat([z_df, z_os], dim=-1))

def mutual_info_loss(joint_logits, neg_logits):
    """
    Binary‐cross-entropy version of the MI estimator.
    """
    bce = nn.BCEWithLogitsLoss()
    loss_joint = bce(joint_logits, torch.ones_like(joint_logits))
    loss_neg   = bce(neg_logits,   torch.zeros_like(neg_logits))
    return loss_joint + loss_neg 

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

def linear_probe_identity(embeddings, labels, max_iter=2000):
    """
    Fit a logistic-regression probe to predict identity.
    Returns (accuracy, fitted_clf).
    """
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, n_jobs=-1)
    )
    probe.fit(embeddings, labels)
    acc = probe.score(embeddings, labels)
    return acc, probe

def tsne_and_ari(
    embeddings, labels, *, max_ids=None, random_state=42, return_fig=True
):
    """
    t-SNE projection + K-means -> Adjusted-Rand-Index, optional figure.
    Returns (ari, fig | None).
    """
    embeddings = np.asarray(embeddings)
    labels     = np.asarray(labels)

    # optional identity slicing
    if max_ids is not None:
        valid = np.unique(labels)[:max_ids]
        keep  = np.isin(labels, valid)
        embeddings, labels = embeddings[keep], labels[keep]

    n_clusters = len(np.unique(labels))

    proj = TSNE(
        n_components=2, perplexity=30, random_state=random_state
    ).fit_transform(embeddings)

    cluster_lbl = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(
        embeddings
    )
    ari = adjusted_rand_score(labels, cluster_lbl)

    if not return_fig:
        return ari, None

    # --- plotting --------------------------------------------------------
    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True   # << use constrained layout
    )

    axs[0].scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab20", alpha=0.7)
    axs[0].set_title("t-SNE · true IDs");  axs[0].axis("off")

    axs[1].scatter(proj[:, 0], proj[:, 1], c=cluster_lbl, cmap="tab20", alpha=0.7)
    axs[1].set_title(f"t-SNE · k-means (k={n_clusters})");  axs[1].axis("off")

    norm = mpl.colors.Normalize(vmin=labels.min(), vmax=labels.max())
    sm   = plt.cm.ScalarMappable(cmap="tab20", norm=norm);  sm.set_array([])
    fig.colorbar(sm, ax=axs.ravel().tolist(), orientation="horizontal", pad=0.04,
                 fraction=0.05, label="Identity")
    return ari, fig