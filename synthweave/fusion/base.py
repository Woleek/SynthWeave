from collections import OrderedDict
import torch
import torch.nn as nn
from ..utils.modules import LazyProjectionMLP, ProjectionMLP, L2NormalizationLayer

from abc import abstractmethod
from typing import Dict, List, Optional


"""
Base fusion module for multimodal feature fusion.

This module provides the base class for all fusion modules in the library.
It implements common functionality for handling multiple modalities and their
projections into a common feature space.
"""


class BaseFusion(nn.Module):
    """Base class for all fusion modules.

    Provides common functionality for handling multiple modalities and their
    projections. All fusion modules should inherit from this class and implement
    the _forward method.

    Attributes:
        modalities (List[str]): List of modality names
        input_dims (Dict[str, int]): Input dimensions for each modality
        proj_dim (Optional[int]): Common projection dimension for all modalities
        projection_layers (nn.ModuleDict): Projection layers for each modality
        dropout (nn.Module): Dropout layer or Identity if dropout is False

    Note:
        Subclasses must implement the _forward method to define the fusion logic.
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
        normalize: bool = True,
    ):
        """Initialize the base fusion module.

        Args:
            modality_keys: List of modality names
            input_dims: Dictionary mapping modality names to their input dimensions
            bias: Whether to include bias in projection layers
            dropout: Dropout probability (if 0, no dropout is applied)
            unify_embeds: Whether to project all modalities to same dimension
            hidden_proj_dim: Hidden dimension for projection MLP
            out_proj_dim: Output dimension for projection MLP
            normalize: Whether to L2 normalize projections

        Note:
            If input_dims is None, lazy layers will be used to infer dimensions.
        """
        super(BaseFusion, self).__init__()

        self.output_dim = output_dim
        self.modalities = modality_keys
        self.bias = bias
        self.dropout_p = dropout_p

        # Set input dimensions
        if input_dims is None:
            input_dims = {mod: None for mod in modality_keys}
        else:
            assert set(modality_keys) == set(
                input_dims.keys()
            ), "Input dimensions must be specified for all modalities."
        self.input_dims = input_dims

        # Set projection layers
        if unify_embeds:
            if out_proj_dim is None:  # Determine output dimension
                assert all(
                    dim is not None for dim in input_dims.values()
                ), "Either specify output dimension or input dimensions for all modalities."
                out_proj_dim = min(dim for dim in input_dims.values())
            self.proj_dim = out_proj_dim
            
            # Set projection dim
            if hidden_proj_dim is not None:
                hidden_proj_dim = [hidden_proj_dim for _ in modality_keys]
            elif self.input_dims and all(
                    dim is not None for dim in input_dims.values()
                ):
                hidden_proj_dim = [dim // 2 for dim in input_dims.values()]
            else:
                hidden_proj_dim = [out_proj_dim // 2 for _ in modality_keys]

            self.projection = nn.ModuleDict(
                {
                    mod: nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        "mlp",
                                        (
                                            ProjectionMLP(
                                                input_dim,
                                                input_dim,
                                                out_proj_dim,
                                                bias=True,
                                                dropout=dropout_p,
                                            )
                                            if input_dim is not None
                                            else LazyProjectionMLP(
                                                input_dim,
                                                out_proj_dim,
                                                bias=True,
                                                dropout=dropout_p,
                                            )
                                        ),
                                    )
                                ),
                                # ("layernorm", nn.LayerNorm(out_proj_dim)),
                            ]
                        )
                    )
                    for idx, (mod, input_dim) in enumerate(self.input_dims.items())
                }
            )
        else:
            self.proj_dim = None

            self.projection = nn.ModuleDict(
                {mod: nn.Identity() for mod in modality_keys}
            )

    def __call__(
        self, embeddings: Dict[str, torch.Tensor], output_projections: bool = False
    ) -> torch.Tensor:
        return self.forward(embeddings, output_projections)

    @abstractmethod
    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Abstract method for implementing specific fusion strategies.

        This method should be implemented by subclasses to define how the
        modality embeddings are fused together.

        Args:
            embeddings (Dict[str, torch.Tensor]): Dictionary of projected embeddings
                for each modality.

        Returns:
            torch.Tensor: The fused embedding representation.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Forward method not implemented!")

    def forward(
        self, embeddings: Dict[str, torch.Tensor], output_projections: bool = False
    ) -> torch.Tensor:
        """Main forward pass of the fusion module.

        Handles projection of embeddings and calls the specific fusion implementation.

        Args:
            embeddings (Dict[str, torch.Tensor]): Dictionary mapping modality names to their
                embeddings. Embeddings can be either:
                - 2D tensors of shape (batch_size, embedding_dim)
                - 3D tensors of shape (batch_size, sequence_length, embedding_dim)

        Returns:
            torch.Tensor: The fused embedding representation.
        """
        # Project embeddings
        proj_embeddings = {}
        for mod in embeddings.keys():
            embed = embeddings[mod]

            if embed.dim() == 3:  # Handle sequence data (B, T, E)
                B, T, E = embed.shape
                embed = embed.reshape(B * T, E)  # Flatten
                embed = self.projection[mod](embed)  # Project
                embed = embed.reshape(B, T, -1)  # Reshape back
                proj_embeddings[mod] = embed

            else:  # (B, E)
                proj_embeddings[mod] = self.projection[mod](embed)

        # Perform fusion
        fused_embedding = self._forward(proj_embeddings)

        # Output
        out = {
            "embedding": fused_embedding,
        }

        if output_projections:
            out.update(
                {
                    f"{mod}_proj": proj_embed
                    for mod, proj_embed in proj_embeddings.items()
                }
            )

        return out
