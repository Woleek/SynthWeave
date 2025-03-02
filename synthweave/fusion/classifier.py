import torch
import torch.nn as nn
from typing import List, Optional

from .. import logger
from .base import BaseFusion
from ..utils.modules import LazyLinearXavier

"""
Classifier-based fusion modules for multimodal feature fusion.

This module implements various classifier-based fusion methods including Majority Voting (MV),
Average Score Fusion (ASF), and Score Fusion (SF). These methods are designed for binary
classification tasks and output predictions directly without requiring additional classifier heads.
"""


class MV(BaseFusion):
    """Majority Voting (MV) fusion module.

    A baseline method that determines the final prediction based on majority vote of
    predictions from each modality. Requires an odd number of modalities to avoid ties.

    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting
    Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103

    Attributes:
        unify_layers: ModuleList of layers to project modalities to same dimension
        clf_layers: ModuleList of binary classifiers for each modality

    Note:
        - Designed specifically for binary classification
        - Requires odd number of modalities
        - Outputs predictions directly (no need for additional classifier)
    """

    def __init__(
        self,
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
    ) -> None:
        """Initialize the MV module.

        Args:
            output_dim: Must be 1 for binary classification
            n_modals: Number of modalities (must be odd)
            dropout: Whether to use dropout
            unify_embeds: Whether to project modalities to same dimension

        Raises:
            ValueError: If n_modals is even or output_dim != 1
        """
        super(MV, self).__init__(dropout, unify_embeds)

        logger.warning(
            "Note that this method outputs predictions instead of vectors and does not require additional classifier head."
        )

        if len(n_modals) % 2 == 0:
            raise ValueError("Majority voting requires an odd number of modalities.")

        if output_dim != 1:
            logger.warning(
                "This method is designed for binary classification. Setting `output_dim` to 1."
            )
            output_dim = 1

        # Unify representations into same dimension
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList(
                [LazyLinearXavier(output_dim) for _ in range(n_modals)]
            )
        else:
            self.unify_layers = nn.ModuleList([nn.Identity() for _ in range(n_modals)])

        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList(
            [
                nn.Sequential(LazyLinearXavier(output_dim), nn.Sigmoid())
                for _ in range(n_modals)
            ]
        )

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the MV module.

        Args:
            embeddings: List of tensors from each modality
                       Shape: [(batch_size, embed_dim), ...]

        Returns:
            torch.Tensor: Binary predictions with shape (batch_size, 1)
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed)
            for unify_layer, embed in zip(self.unify_layers, embeddings)
        ]

        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, proj_embeds)]

        # Binarize predictions
        preds = [torch.round(logit) for logit in logits]

        # Majority voting
        votes = torch.stack(preds, dim=-1)
        results = torch.mode(votes, dim=-1).values.float()

        return results.view(-1, 1)


class ASF(BaseFusion):
    """Average Score Fusion (ASF) module.

    A baseline method that averages the prediction scores from each modality and applies
    a threshold to determine the final binary prediction.

    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting
    Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103

    Attributes:
        unify_layers: ModuleList of layers to project modalities to same dimension
        clf_layers: ModuleList of binary classifiers for each modality
        threshold: Threshold for binary classification (default: 0.5)

    Note:
        - Designed specifically for binary classification
        - Outputs predictions directly (no need for additional classifier)
    """

    def __init__(
        self,
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
    ) -> None:
        """Initialize the ASF module.

        Args:
            output_dim: Must be 1 for binary classification
            n_modals: Number of modalities
            dropout: Whether to use dropout
            unify_embeds: Whether to project modalities to same dimension
        """
        super(ASF, self).__init__(dropout, unify_embeds)

        logger.warning(
            "Note that this method outputs predictions instead of vectors and does not require additional classifier head."
        )

        if output_dim != 1:
            logger.warning(
                "This method is designed for binary classification. Setting `output_dim` to 1."
            )
            output_dim = 1

        # Unify representations into same dimension
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList(
                [LazyLinearXavier(output_dim) for _ in range(n_modals)]
            )
        else:
            self.unify_layers = nn.ModuleList([nn.Identity() for _ in range(n_modals)])

        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList(
            [
                nn.Sequential(LazyLinearXavier(output_dim), nn.Sigmoid())
                for _ in range(n_modals)
            ]
        )

        # Threshold for binarization
        self.threshold = 0.5

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the ASF module.

        Args:
            embeddings: List of tensors from each modality
                       Shape: [(batch_size, embed_dim), ...]

        Returns:
            torch.Tensor: Binary predictions with shape (batch_size, 1)
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed)
            for unify_layer, embed in zip(self.unify_layers, embeddings)
        ]

        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, proj_embeds)]

        # Average scores
        avg_scores = torch.stack(logits, dim=-1).mean(dim=-1)

        # Binarize predictions
        results = (avg_scores > self.threshold).float()

        return results.view(-1, 1)


class SF(BaseFusion):
    """Score Fusion (SF) module.

    A baseline method that combines prediction scores from each modality using
    a learnable fusion layer to determine the final binary prediction.

    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting
    Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103

    Attributes:
        unify_layers: ModuleList of layers to project modalities to same dimension
        clf_layers: ModuleList of binary classifiers for each modality
        score_fusion_layer: Learnable layer for combining modality scores

    Note:
        - Designed specifically for binary classification
        - Outputs predictions directly (no need for additional classifier)
    """

    def __init__(
        self,
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True,
    ) -> None:
        """Initialize the SF module.

        Args:
            output_dim: Must be 1 for binary classification
            n_modals: Number of modalities
            dropout: Whether to use dropout
            unify_embeds: Whether to project modalities to same dimension
        """
        super(SF, self).__init__(dropout, unify_embeds)

        logger.warning(
            "Note that this method outputs predictions instead of vectors and does not require additional classifier head."
        )

        if output_dim != 1:
            logger.warning(
                "This method is designed for binary classification. Setting `output_dim` to 1."
            )
            output_dim = 1

        # Unify representations into same dimension
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList(
                [LazyLinearXavier(output_dim) for _ in range(n_modals)]
            )
        else:
            self.unify_layers = nn.ModuleList([nn.Identity() for _ in range(n_modals)])

        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList(
            [LazyLinearXavier(output_dim) for _ in range(n_modals)]
        )

        self.score_fusion_layer = nn.Sequential(
            LazyLinearXavier(output_dim), nn.Sigmoid()
        )

    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass for the SF module.

        Args:
            embeddings: List of tensors from each modality
                       Shape: [(batch_size, embed_dim), ...]

        Returns:
            torch.Tensor: Binary predictions with shape (batch_size, 1)
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed)
            for unify_layer, embed in zip(self.unify_layers, embeddings)
        ]

        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, proj_embeds)]

        # Score fusion
        scores = torch.stack(logits, dim=-1)
        preds = self.score_fusion_layer(scores)

        # Binarize predictions
        results = torch.round(preds)

        return results.view(-1, 1)
