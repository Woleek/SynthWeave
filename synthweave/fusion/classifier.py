import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .. import logger
from .base import BaseFusion

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
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout_p: float = 0.1,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the MV module.

        Args:
            output_dim: Must be 1 for binary classification
            modality_keys: List of modality names to be fused (must be odd length)
            input_dims: Dictionary mapping modality names to input dimensions
            bias: Whether to include bias in linear layers
            dropout_p: Dropout probability
            unify_embeds: Whether to project modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection layers
            out_proj_dim: Output dimension for projection layers
            normalize_proj: Whether to apply L2 normalization after projection

        Raises:
            ValueError: If number of modalities is even
        """
        super(MV, self).__init__(
            output_dim=1,
            modality_keys=modality_keys,
            input_dims=input_dims,
            bias=bias,
            dropout_p=dropout_p,
            unify_embeds=unify_embeds,
            hidden_proj_dim=hidden_proj_dim,
            out_proj_dim=out_proj_dim,
            normalize=normalize_proj,
        )

        logger.warning(
            "Note that this method outputs predictions instead of vectors and does not require additional classifier head."
        )

        if len(self.modalities) % 2 == 0:
            raise ValueError("Majority voting requires an odd number of modalities.")

        if output_dim != 1:
            logger.warning(
                "This method is designed for binary classification. Setting `output_dim` to 1."
            )

        self.clf_layers = nn.ModuleDict(
            {
                modal: nn.Sequential(nn.LazyLinear(1), nn.Sigmoid())
                for modal in self.modalities
            }
        )

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the MV module.

        Args:
            embeddings: Dictionary mapping modality names to projected tensors
                       Shape per modality: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Binary predictions with shape (batch_size, 1)
        """
        # Get probabilities from each modality
        logits = [self.clf_layers[modal](embeddings[modal]) for modal in self.modalities]

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
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout_p: float = 0.1,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the ASF module.

        Args:
            output_dim: Must be 1 for binary classification
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to input dimensions
            bias: Whether to include bias in linear layers
            dropout_p: Dropout probability
            unify_embeds: Whether to project modalities to same dimension
        """
        super(ASF, self).__init__(
            output_dim=1,
            modality_keys=modality_keys,
            input_dims=input_dims,
            bias=bias,
            dropout_p=dropout_p,
            unify_embeds=unify_embeds,
            hidden_proj_dim=hidden_proj_dim,
            out_proj_dim=out_proj_dim,
            normalize=normalize_proj,
        )

        logger.warning(
            "Note that this method outputs predictions instead of vectors and does not require additional classifier head."
        )

        if output_dim != 1:
            logger.warning(
                "This method is designed for binary classification. Setting `output_dim` to 1."
            )
        self.clf_layers = nn.ModuleDict(
            {
                modal: nn.Sequential(nn.LazyLinear(1), nn.Sigmoid())
                for modal in self.modalities
            }
        )

        # Threshold for binarization
        self.threshold = 0.5

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the ASF module.

        Args:
            embeddings: Dictionary mapping modality names to projected tensors
                       Shape per modality: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Binary predictions with shape (batch_size, 1)
        """
        # Get probabilities from each modality
        logits = [self.clf_layers[modal](embeddings[modal]) for modal in self.modalities]

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
        modality_keys: List[str],
        input_dims: Optional[Dict[str, int]] = None,
        bias: bool = True,
        dropout_p: float = 0.1,
        unify_embeds: bool = True,
        hidden_proj_dim: Optional[int] = None,
        out_proj_dim: Optional[int] = None,
        normalize_proj: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the SF module.

        Args:
            output_dim: Must be 1 for binary classification
            modality_keys: List of modality names to be fused
            input_dims: Dictionary mapping modality names to input dimensions
            bias: Whether to include bias in linear layers
            dropout_p: Dropout probability
            unify_embeds: Whether to project modalities to same dimension
        """
        super(SF, self).__init__(
            output_dim=1,
            modality_keys=modality_keys,
            input_dims=input_dims,
            bias=bias,
            dropout_p=dropout_p,
            unify_embeds=unify_embeds,
            hidden_proj_dim=hidden_proj_dim,
            out_proj_dim=out_proj_dim,
            normalize=normalize_proj,
        )

        logger.warning(
            "Note that this method outputs predictions instead of vectors and does not require additional classifier head."
        )

        if output_dim != 1:
            logger.warning(
                "This method is designed for binary classification. Setting `output_dim` to 1."
            )
        self.clf_layers = nn.ModuleDict(
            {modal: nn.LazyLinear(1) for modal in self.modalities}
        )

        self.score_fusion_layer = nn.Sequential(
            nn.LazyLinear(1), nn.Sigmoid()
        )

    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the SF module.

        Args:
            embeddings: Dictionary mapping modality names to projected tensors
                       Shape per modality: (batch_size, embed_dim)

        Returns:
            torch.Tensor: Binary predictions with shape (batch_size, 1)
        """
        # Get probabilities from each modality
        logits = [self.clf_layers[modal](embeddings[modal]) for modal in self.modalities]

        # Score fusion
        scores = torch.cat(logits, dim=1)
        preds = self.score_fusion_layer(scores)

        # Binarize predictions
        results = torch.round(preds)

        return results.view(-1, 1)
