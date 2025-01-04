import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Minimizes the distance between similar pairs and maximizes the distance between dissimilar pairs."""
    def __init__(self, margin: float = 0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss between a set of embeddings.
        """
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2) + 1e-8  # Shape: (batch_size, batch_size)

        # Create label match matrix
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # Shape: (batch_size, batch_size)

        # Positive pairs: same labels
        pos_loss = (label_matrix) * (distances ** 2)

        # Negative pairs: different labels
        neg_loss = (1 - label_matrix) * torch.clamp(self.margin - distances, min=0) ** 2
        
        # Compute loss
        loss = (pos_loss + neg_loss).sum() / embeddings.size(0)
        
        return loss

    
class ContrastiveLossWithBatchBalancing(nn.Module):
    """
    Computes the contrastive loss, normalized by the number of valid pairs to address class imbalance.
    """
    def __init__(self, margin: float = 0.2):
        super(ContrastiveLossWithBatchBalancing, self).__init__()
        self.margin = margin
       
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2) + 1e-8  # Shape: (batch_size, batch_size)

        # Create label match matrix
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # Shape: (batch_size, batch_size)

        # Positive pairs: same labels
        pos_loss = (label_matrix) * (distances ** 2)

        # Negative pairs: different labels
        neg_loss = (1 - label_matrix) * torch.clamp(self.margin - distances, min=0) ** 2
        
        # Normalize by the number of valid pairs
        num_pos_pairs = label_matrix.sum()
        num_neg_pairs = (1 - label_matrix).sum()
        
        # Compute loss
        pos_loss = pos_loss.sum() / (num_pos_pairs + 1e-8)
        neg_loss = neg_loss.sum() / (num_neg_pairs + 1e-8)
        loss = pos_loss + neg_loss
        
        return loss