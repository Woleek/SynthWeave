import torch
import torch.nn as nn
from ...utils.modules import LinearXavier
        
class SelfAttention(nn.Module):
    """
    Self-Attention (SelfAtt) sub-layer.
    """
    def __init__(self, modal_dim):
        super(SelfAttention, self).__init__()
        self.Wq = LinearXavier(modal_dim, modal_dim)
        self.Wk = LinearXavier(modal_dim, modal_dim)
        self.Wv = LinearXavier(modal_dim, modal_dim)
        self.scale = 1 / (modal_dim ** 0.5)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Z):
        Q = self.Wq(Z)
        K = self.Wk(Z)
        V = self.Wv(Z)
        Z_hat = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale) @ V
        return Z_hat

class FeedForward(nn.Module):
    """
    Feed-Forward (FF) sub-layer.
    """
    def __init__(self, modal_dim, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = LinearXavier(modal_dim, d_ff)
        self.linear2 = LinearXavier(d_ff, modal_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))
    
class BiCroAttention(nn.Module):
    """
    Bi-Directional Cross-Attention (BiCroAtt) block.
    """
    def __init__(self, modal_dim):
        super(BiCroAttention, self).__init__()
        self.Wq = LinearXavier(modal_dim, modal_dim)
        self.Wk = LinearXavier(modal_dim, modal_dim)
        self.Wv = LinearXavier(modal_dim, modal_dim)
        self.scale = 1 / (modal_dim ** 0.5)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, P_i: torch.Tensor, *P_others: list[torch.Tensor]) -> torch.Tensor:
        """
        Compute bi-directional cross-attention for the i-th modality.
        """
        Qi = self.Wq(P_i)
        Ki = self.Wk(P_i)
        Vi = self.Wv(P_i)

        attended_inputs = []
        for P_j in P_others:  # Iterate over all other modalities
            Qj = self.Wq(P_j)
            Kj = self.Wk(P_j)
            Vj = self.Wv(P_j)
            
            # Compute cross-attention between current modality and each other modality     
            attended_inputs.append(self.softmax(torch.matmul(Qi, Kj.transpose(-2, -1)) * self.scale) @ Vj)

        return torch.cat(attended_inputs, dim=-1)
    
class MMDBlock(nn.Module):
    """
    Single block of the Multi-Modal Joint-Decoder (MMD).
    Implements BiCroAtt, SelfAtt, and FeedForward with residual connections and LayerNorm.
    """
    def __init__(self, modality_dim) -> None:
        """
        Initialize the MMDBlock.
        """
        super(MMDBlock, self).__init__()
        
        # BiCroAtt: Bi-Directional Cross-Attention
        self.bi_cro_att = BiCroAttention(modality_dim)
        
        # SelfAtt: Self-Attention
        self.self_att = SelfAttention(modality_dim)
        
        # FeedForward: Feed-Forward layer
        self.ff = FeedForward(modality_dim, modality_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(modality_dim)
        self.ln2 = nn.LayerNorm(modality_dim)
        self.ln3 = nn.LayerNorm(modality_dim)

    def forward(self, proj_embeds: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass for a single MMDBlock.
        """
        
        # BiCroAtt with residual connection
        attended_embeds = [
            self.ln1(proj_embeds[i] + self.bi_cro_att(proj_embeds[i], *[proj_embeds[j] for j in range(len(proj_embeds)) if j != i]))
            for i in range(len(proj_embeds))
        ]

        # SelfAtt with residual connection
        refined_embeds = [
            self.ln2(attended_embeds[i] + self.self_att(attended_embeds[i]))
            for i in range(len(attended_embeds))
        ]

        # FeedForward with residual connection
        refined_embeds = [
            self.ln3(refined_embeds[i] + self.ff(refined_embeds[i]))
            for i in range(len(refined_embeds))
        ]

        return refined_embeds