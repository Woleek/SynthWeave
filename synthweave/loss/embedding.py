import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Embedding loss functions for deep metric learning.

This module provides loss functions designed for learning discriminative embeddings,
including contrastive loss variants that help create more robust and separable
feature representations in the embedding space.
"""


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative embeddings.

    Minimizes the distance between similar pairs and maximizes the distance between
    dissimilar pairs up to a margin. This helps create embeddings where similar
    samples are close together and dissimilar samples are far apart.

    Based on: "Dimensionality Reduction by Learning an Invariant Mapping"
    Source: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    Attributes:
        margin: Minimum distance margin between dissimilar pairs
    """

    def __init__(self, margin: float = 0.2):
        """Initialize the contrastive loss module.

        Args:
            margin: Minimum distance margin between dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the contrastive loss between a set of embeddings.

        Args:
            embeddings: Feature embeddings of shape (batch_size, embed_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value

        Process:
            1. Compute pairwise Euclidean distances between embeddings
            2. Create label match matrix to identify similar/dissimilar pairs
            3. Compute loss for positive pairs (similar labels)
            4. Compute loss for negative pairs (different labels)
            5. Combine and normalize the losses
        """
        # Compute pairwise distances
        distances = (
            torch.cdist(embeddings, embeddings, p=2) + 1e-8
        )  # Shape: (batch_size, batch_size)

        # Create label match matrix
        label_matrix = (
            labels.unsqueeze(0) == labels.unsqueeze(1)
        ).float()  # Shape: (batch_size, batch_size)

        # Positive pairs: same labels
        pos_loss = (label_matrix) * (distances**2)

        # Negative pairs: different labels
        neg_loss = (1 - label_matrix) * torch.clamp(self.margin - distances, min=0) ** 2

        # Compute loss
        loss = (pos_loss + neg_loss).sum() / embeddings.size(0)

        return loss


class ContrastiveLossWithBatchBalancing(nn.Module):
    """Contrastive loss with batch-wise pair balancing.

    An enhanced version of ContrastiveLoss that normalizes the loss by the number
    of valid pairs in each category (similar/dissimilar). This helps address class
    imbalance issues and provides more stable training.

    Attributes:
        margin: Minimum distance margin between dissimilar pairs
    """

    def __init__(self, margin: float = 0.2):
        """Initialize the balanced contrastive loss module.

        Args:
            margin: Minimum distance margin between dissimilar pairs
        """
        super(ContrastiveLossWithBatchBalancing, self).__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the balanced contrastive loss between a set of embeddings.

        Args:
            embeddings: Feature embeddings of shape (batch_size, embed_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value

        Process:
            1. Compute pairwise Euclidean distances between embeddings
            2. Create label match matrix to identify similar/dissimilar pairs
            3. Compute loss for positive pairs (similar labels)
            4. Compute loss for negative pairs (different labels)
            5. Normalize each loss term by its number of pairs
            6. Sum the normalized losses
        """
        # Compute pairwise distances
        distances = (
            torch.cdist(embeddings, embeddings, p=2) + 1e-8
        )  # Shape: (batch_size, batch_size)

        # Create label match matrix
        label_matrix = (
            labels.unsqueeze(0) == labels.unsqueeze(1)
        ).float()  # Shape: (batch_size, batch_size)

        # Positive pairs: same labels
        pos_loss = (label_matrix) * (distances**2)

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
