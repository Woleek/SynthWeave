
import math
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kl(p, q):
    """KL(p‖q) for *batched* row-stochastic tensors (B, K)."""
    p_clamped = p.clamp_min(1e-7)
    q_clamped = q.clamp_min(1e-7)
    return (p_clamped * (p_clamped.log() - q_clamped.log())).sum(dim=-1)  # (B,)

def jsd(p, q):
    """Symmetric Jensen-Shannon divergence, returns shape (B,)."""
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

class JSDPredictionLoss(nn.Module):
    """
    Computes 𝓛_JSD^pred with sign-flip, *and* contains its own 2-class
    heads for audio / video so you don’t have to declare them elsewhere.
    """
    def __init__(self, dim: int, p_drop: float = 0.3):
        super().__init__()
        self.audio_cls = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.LeakyReLU(),
            nn.Dropout(p_drop),       nn.Linear(dim // 2, 2)
        )
        self.video_cls = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.LeakyReLU(),
            nn.Dropout(p_drop),       nn.Linear(dim // 2, 2)
        )

    def forward(
        self,
        fa: torch.Tensor,            # (B, dim) audio_proj
        fv: torch.Tensor,            # (B, dim) video_proj
        la: torch.Tensor,            # (B,) 0/1  forged flag (audio)
        lv: torch.Tensor             # (B,) 0/1  forged flag (video)
    ) -> torch.Tensor:
        pa = F.softmax(self.audio_cls(fa), dim=-1)   # (B, 2)
        pv = F.softmax(self.video_cls(fv), dim=-1)   # (B, 2)

        same = (la == lv)            # bool mask (B,)
        d    = jsd(pa, pv)           # (B,)

        d = torch.where(same, d, -d) # sign-flip for mismatched labels
        return d.mean()

class JSDFeatureLoss(nn.Module):
    """
    𝓛_JSD^feat with:
      • L2-normalisation
      • temperature τ (default 0.07)
      • same sign-flip rule
    """
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        fa: torch.Tensor,            # (B, dim) audio_proj
        fv: torch.Tensor,            # (B, dim) video_proj
        la: torch.Tensor,            # (B,)
        lv: torch.Tensor             # (B,)
    ) -> torch.Tensor:
        fa_s = F.softmax(F.normalize(fa, dim=-1) / self.tau, dim=-1)
        fv_s = F.softmax(F.normalize(fv, dim=-1) / self.tau, dim=-1)

        same = (la == lv)
        d    = jsd(fa_s, fv_s)
        d = torch.where(same, d, -d)
        return d.mean()


    
class CenterClusterLoss(nn.Module):
    """
    Computes the Audio‐Visual Center Cluster Loss (ℒ_cc) for a batch, using K learnable centers
    and only the top P% hardest samples in each class, plus a small repulsion penalty among centers.
    
    Maintains K learnable center vectors C = {c_j ∈ ℝ^D | j = 1..K}. For each sample i, let
      d_i² = min_{j=1..K} ||cls_i^g − c_j||².

    Then (with hard mining):
      Let Ω_r = indices of real samples, Ω_f = indices of forged samples.
      
      For real:
        • Compute all d_i² for i∈Ω_r.
        • Sort descending and keep top k_r = ⌈P·|Ω_r|⌉ largest distances.
        • ℒ_r = (1 / (2·k_r)) Σ_{i in top‐k_r} d_i².

      For forged:
        • Compute all d_i² for i∈Ω_f.
        • Sort ascending and keep top k_f = ⌈P·|Ω_f|⌉ smallest distances.
        • avg_f = (1 / (2·k_f)) Σ_{i in top‐k_f} d_i².
        • ℒ_f = min(avg_f, γ₂).

      Total: ℒ_cc = ℒ_r − ℒ_f.

    We now add:
      For all pairs of centers (c_i, c_j), i < j:
        d_{ij} = ||c_i − c_j|| .
        hinge_{ij} = max(0, center_margin − d_{ij}).
      Center‐repulsion penalty:
        ℒ_ctr = (1 / (#pairs)) Σ_{i<j} hinge_{ij}.
    
    Final total = ℒ_cc + λ_center · ℒ_ctr.

    If Ω_r or Ω_f is empty in the batch, that term is treated as zero.
    """

    def __init__(
        self,
        embed_dim: int,
        num_centers: int = 1,
        top_p: float = 1.0,
        gamma2: float = 0.25,
        center_margin: float = 1.0,
        lambda_center: float = 1e-3,
        eps: float = 1e-6
    ):
        """
        Args:
            embed_dim:      Dimensionality D of each CLS embedding.
            num_centers:    Number of learnable centers K (default=1).
            top_p:          Fraction P ∈ (0, 1] of hardest samples to use (default=1.0).
            gamma2:         Margin γ₂ for forged term (default=0.25).
            center_margin:  Minimum desired distance between any two centers (default=1.0).
            lambda_center:  Weight for the center‐repulsion penalty (default=1e-3).
            eps:            Small value to avoid division by zero (default=1e-6).
        """
        super().__init__()
        if num_centers < 1:
            raise ValueError(f"num_centers must be ≥ 1, got {num_centers}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        self.embed_dim = embed_dim
        self.num_centers = num_centers
        self.top_p = top_p
        self.gamma2 = gamma2
        self.center_margin = center_margin
        self.lambda_center = lambda_center
        self.eps = eps

        # Learnable centers: shape (K, D), initialized to zero
        self.centers = nn.Parameter(torch.zeros(num_centers, embed_dim))
        
    def initialize_centers_from(
        self,
        cls_global: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = None
    ):
        """
        One‐time initialization of `self.centers` using KMeans on the real‐sample embeddings.

        Args:
            cls_global: FloatTensor of shape (B, D), the [CLS] embeddings for a batch.
            labels:     Tensor of shape (B,), with 0=real, 1=forged.
            device:     (optional) where to place the initialized centers.
                        If None, we infer from cls_global.device.
        """
        # Only run if num_centers > 1; if K=1, keep zero or random
        if self.num_centers < 2:
            return

        if device is None:
            device = cls_global.device

        # 1) pick only real‐class (label==0) embeddings
        real_mask = (labels == 0)
        real_embeddings = cls_global[real_mask].detach().cpu().numpy()  # (n_real, D)

        if real_embeddings.shape[0] < self.num_centers:
            # Fallback: take random samples as initial centers if not enough reals
            # We will randomly sample K distinct real indices (or just random noise)
            rand_idx = torch.randperm(cls_global.size(0))[: self.num_centers]
            init = cls_global[rand_idx].detach().cpu().numpy()
        else:
            # 2) run KMeans on these real embeddings
            kmeans = KMeans(n_clusters=self.num_centers, random_state=42, n_init=10)
            kmeans.fit(real_embeddings)
            init = kmeans.cluster_centers_  # shape (K, D)

        # 3) overwrite self.centers with those centroids
        centroids = torch.from_numpy(init).to(device).float()  # (K, D)
        with torch.no_grad():
            self.centers.data.copy_(centroids)

    def forward(
        self,
        cls_global: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cls_global: Tensor of shape (B, D), the [CLS] embeddings for the batch.
            labels:     Binary Tensor of shape (B,), where 0 = real, 1 = forged.
        Returns:
            A scalar Tensor containing ℒ_cc + (λ_center · ℒ_ctr).
        """
        device = cls_global.device
        y = labels.float()  # shape (B,)

        B, D = cls_global.shape
        # 1) Compute squared distances to each of the K centers:
        #    cls_global: (B, D) -> (B, 1, D)
        #    centers:    (K, D) -> (1, K, D)
        diff = cls_global.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, K, D)
        dist2_all = torch.sum(diff * diff, dim=2)                  # (B, K)

        # 2) For each sample i, take d_i² = min_j dist2_all[i, j].
        dist2, _ = torch.min(dist2_all, dim=1)  # (B,)

        # Masks for real vs. forged
        mask_real = (y == 0.0)
        mask_forged = (y == 1.0)

        # Count how many real/forged samples we have
        num_real = int(mask_real.sum().item())
        num_forged = int(mask_forged.sum().item())

        # ---------- Real term (hard‐mining P% of real, i.e. largest distances) ----------
        if num_real > 0:
            real_dist2 = dist2[mask_real]  # shape (num_real,)
            # k_real = ceil(P · |Ω_r|), at least 1
            k_real = max(1, int(math.ceil(self.top_p * num_real)))
            if k_real < num_real:
                # pick the k_real largest distances
                hardest_real_vals = torch.topk(real_dist2, k_real, largest=True).values
            else:
                # use all real samples
                hardest_real_vals = real_dist2
            real_loss = hardest_real_vals.sum() / (2.0 * (k_real + self.eps))
        else:
            real_loss = torch.tensor(0.0, device=device)

        # ---------- Forged term (hard‐mining P% of forged, i.e. smallest distances) ----------
        if num_forged > 0:
            forged_dist2 = dist2[mask_forged]  # shape (num_forged,)
            # k_forged = ceil(P · |Ω_f|), at least 1
            k_forged = max(1, int(math.ceil(self.top_p * num_forged)))
            if k_forged < num_forged:
                # pick the k_forged smallest distances
                hardest_forged_vals = torch.topk(forged_dist2, k_forged, largest=False).values
            else:
                # use all forged samples
                hardest_forged_vals = forged_dist2
            avg_forged = hardest_forged_vals.sum() / (2.0 * (k_forged + self.eps))
            forged_term = torch.minimum(avg_forged, torch.tensor(self.gamma2, device=device))
        else:
            forged_term = torch.tensor(0.0, device=device)

        # Base cluster‐loss: ℒ_cc = ℒ_r − ℒ_f
        cc_loss = real_loss - forged_term

        # ---------- Center‐repulsion penalty to prevent collapse ----------
        # Only makes sense if K > 1
        if self.num_centers > 1 and self.lambda_center > 0.0:
            # Compute pairwise Euclidean distances among centers:
            # centers: (K, D)
            c = self.centers  # (K, D)
            # Compute squared distances: (K, K)
            c_diff = c.unsqueeze(0) - c.unsqueeze(1)  # (K, K, D)
            c_dist2_mat = torch.sum(c_diff * c_diff, dim=2)  # (K, K)

            # We only need i < j
            idx_i, idx_j = torch.triu_indices(self.num_centers, self.num_centers, offset=1)
            # dist² for each pair (i < j):
            dist2_pairs = c_dist2_mat[idx_i, idx_j]  # (num_pairs,)
            # Convert to Euclidean distance (add eps inside sqrt for numerical stability):
            dist_pairs = torch.sqrt(dist2_pairs + self.eps)  # (num_pairs,)

            # Hinge penalty = max(0, center_margin − dist_pairs)
            hinge_vals = F.relu(self.center_margin - dist_pairs)  # (num_pairs,)

            # Average over all pairs:
            num_pairs = dist_pairs.size(0)  # = K * (K − 1) / 2
            ctr_repulsion = hinge_vals.sum() / (num_pairs + self.eps)

            # Weighted by λ_center
            repulsion_loss = self.lambda_center * ctr_repulsion
        else:
            repulsion_loss = torch.tensor(0.0, device=device)

        total_loss = cc_loss + repulsion_loss
        return total_loss

 
def cross_modal_contrastive_loss(
    cls_audio: torch.Tensor,
    cls_visual: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2
) -> torch.Tensor:
    """
    Computes the Cross‐Modal Contrastive Loss (ℒ_cmc) for a batch.

    Args:
        cls_audio:   Tensor of shape (B, D), the [CLS] embeddings from the audio branch.
        cls_visual:  Tensor of shape (B, D), the [CLS] embeddings from the visual branch.
        labels:      Binary Tensor of shape (B,), where 0 = both real (RVRA), 1 = any forged.
        margin:      Margin γ₁ (default=0.2).

    Returns:
        A scalar Tensor containing ℒ_cmc = (1/N) Σ_i [ (1−y_i)·(1−sim) + y_i·max(sim−γ₁, 0 ) ].
    """
    # Ensure labels are float (0.0 or 1.0)
    y = labels.float()

    # Cosine similarity between cls_audio and cls_visual: shape (B,)
    sim = F.cosine_similarity(cls_audio, cls_visual, dim=-1)  # ∈ [−1, +1]

    # For real samples (y=0): term_0 = 1 - sim
    term_real = (1.0 - sim) * (1.0 - y)  # zeroed out when y=1

    # For forged samples (y=1): term_1 = max(sim - margin, 0)
    diff = sim - margin
    zero = torch.zeros_like(diff)
    term_forged = torch.maximum(diff, zero) * y

    loss = term_real + term_forged
    return loss.mean()    

def cross_modality_regularization(
    x_a: torch.Tensor,           # (B, d)  audio branch
    x_v: torch.Tensor,           # (B, d)  visual branch
    y_c: torch.Tensor,           # (B,)    0 = genuine pair, 1 = forged / mismatched
) -> torch.Tensor:
    """
    Eq.(1) in the MRDF paper.
    """
    sim = F.cosine_similarity(x_a, x_v, dim=-1)           # (B,)
    real_term   = (1.0 - sim) * (1.0 - y_c)               # genuine pairs
    forged_term = F.relu(sim) * y_c                       # fake / mismatched
    return (real_term + forged_term).mean()

class WithinModalityCELoss(nn.Module):
    """
    CE variant (authors say this works best). One tiny classifier per modality.
    """
    def __init__(self, dim: int, num_classes: int = 2, p_drop: float = 0.3):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.LeakyReLU(),
            nn.Dropout(p_drop),        nn.Linear(dim // 2, num_classes)
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat: torch.Tensor, y: torch.Tensor):
        logits = self.cls(feat)               # (B, K)
        return self.ce(logits, y.long())

class WithinModalityMarginLoss(nn.Module):
    """
    Margin-based variant (Eq.(2) in the excerpt) with hyper-parameter α = 0.
    """
    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat: torch.Tensor, y: torch.Tensor):
        feat = F.normalize(feat, dim=-1)                      # cosine space
        sim  = torch.matmul(feat, feat.t())                  # (B, B)

        same_class   = (y.unsqueeze(0) == y.unsqueeze(1))    # (B, B)
        diff_class   = ~same_class

        pos_loss = (1.0 - sim)[same_class].sum()
        neg_loss = F.relu(sim - self.margin)[diff_class].sum()

        denom = max(1, feat.size(0) ** 2)                    # avoid /0
        return (pos_loss + neg_loss) / denom
