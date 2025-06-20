import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fusion.base import BaseFusion
from typing import Any, Dict, Optional, Callable, Tuple, Union, Mapping, List

"""
Base pipeline module for multimodal processing.

This module implements the base pipeline class that handles the complete workflow
of multimodal processing, from input preprocessing through feature extraction,
fusion, and final task-specific processing.
"""


class BasePipeline(nn.Module):
    """Base class for multimodal processing pipelines.

    Implements a complete pipeline workflow in the following sequence:
    1. Uni-modal inputs
    2. Preprocessing
    3. Feature Extraction
    4. Fusion
    5. Head (task specific)
    6. Output

    Attributes:
        feature_extractors (nn.ModuleDict): Feature extraction models for each modality
        fusion (BaseFusion): Fusion module for combining modality features
        head (nn.Module): Task-specific head network
        processors (Mapping[str, Callable]): Preprocessing functions for each modality
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        fusion: BaseFusion,
        head: Optional[nn.Module] = None,
        processors: Optional[Mapping[str, Callable[..., torch.Tensor]]] = None,
        freeze_backbone: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            models: Dictionary mapping modality names to their feature extractors
            fusion: Fusion module or name of fusion method to use
            head: Optional task-specific head network
            processors: Optional dictionary of preprocessing functions
            freeze_backbone: Whether to freeze feature extractor parameters
        """
        super(BasePipeline, self).__init__()

        self.feature_extractors = nn.ModuleDict(models)
        if freeze_backbone:
            self._freeze_backbone()
        self.fusion = fusion
        self.head = head
        self.processors = processors

    def __call__(self, inputs: Dict[str, Any], **kwargs) -> torch.Tensor:
        """Forward pass alias.

        Args:
            inputs: Dictionary of inputs for each modality

        Returns:
            torch.Tensor: Pipeline outputs
        """
        return self.forward(inputs, **kwargs)

    def _freeze_backbone(self) -> None:
        """Freeze parameters of the feature extractors.

        Sets requires_grad=False for all parameters in the feature extraction models
        to prevent updating during training.
        """
        for model in self.feature_extractors.values():
            for param in model.parameters():
                param.requires_grad = False

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Preprocess the raw inputs for each modality.

        Args:
            inputs: Dictionary of raw inputs for each modality

        Returns:
            Dict[str, torch.Tensor]: Preprocessed inputs

        Note:
            If no processors are defined, returns inputs unchanged.
        """
        if self.processors is not None:
            # for modality, processor in self.processors.items():
            #     inputs[modality] = processor(*inputs[modality])
            processed = {
                modality: processor(*inputs[modality])
                for modality, processor in self.processors.items()
            }
            return processed
        else:
            return inputs

    def extract_features(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract features from each modality.

        Args:
            inputs: Dictionary of preprocessed inputs

        Returns:
            Dict[str, torch.Tensor]: Extracted features for each modality
        """
        # for modality, model in self.feature_extractors.items():
        #     inputs[modality] = model(inputs[modality])
        extracted = {
            modality: model(inputs[modality])
            for modality, model in self.feature_extractors.items()
        }
        return extracted

    def downstream_pass(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply task-specific head to fused features.

        Args:
            inputs: Dictionary containing fused embeddings

        Returns:
            torch.Tensor: Task-specific outputs

        Note:
            If no head is defined, returns inputs unchanged.
        """
        if self.head is not None:
            inputs["logits"] = self.head(inputs["embedding"])
        return inputs

    def forward(
        self,
        inputs: Dict[str, Any],
        output_feats: bool = False,
        output_projections: bool = False,
    ) -> torch.Tensor:
        """Complete forward pass through the pipeline.

        Args:
            inputs: Dictionary of raw inputs for each modality
            output_feats: Whether to include individual modality features in output

        Returns:
            torch.Tensor: Pipeline outputs including task-specific predictions

        Process:
            1. Preprocess raw inputs
            2. Extract features from each modality
            3. Fuse features from all modalities
            4. Apply task-specific head
            5. Optionally include individual features in output
        """
        # Preprocess inputs
        inputs = self.preprocess(inputs)

        # Extract embeddings from each modality
        feats = self.extract_features(inputs)

        # Fuse embeddings into one vector
        output: dict = self.fusion(
            {modality: F.normalize(feats[modality], p=2, dim=1) for modality in self.fusion.modalities},
            output_projections=output_projections,
        )

        if output_feats:
            output.update(feats)

        # Pass the fused embeddings to the head
        output = self.downstream_pass(output)

        return output
