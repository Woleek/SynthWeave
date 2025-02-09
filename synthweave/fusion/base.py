import torch
import torch.nn as nn
from ..utils.modules import LazyProjectionMLP, ProjectionMLP, L2NormalizationLayer

from abc import abstractmethod
from typing import Dict, List, Optional

class BaseFusion(nn.Module):
    """
    Base class for all fusion modules.
    """
    def __init__(
            self,
            modality_keys: List[str],
            input_dims: Optional[Dict[str, int]] = None,
            bias: bool = True,
            dropout: float = 0.5,
            
            unify_embeds: bool = True,
            hidden_proj_dim: Optional[int] = None,
            out_proj_dim: Optional[int] = None,
            normalize_proj: bool = True
        ):
        """
        Initializes the base fusion module.
        
        Args:
            modality_keys: List of keys (names) for modalities.
            input_dims: Dictionary of input dimensions for each modality (if not specified will use LazyLinear layers).
            bias: Whether to use bias in linear layers.
            
            unify_embeds: Whether to project embeddings to common dimensions.
            hidden_proj_dim: Hidden dimension for projection MLPs (if None will use output dimension).
            out_proj_dim: Output dimension for projection MLPs (if None will use smaller of input dimensions).
        """
        super(BaseFusion, self).__init__()
        
        self.modalities = modality_keys
        self.bias = bias
        self.dropout = dropout
        
        # Set input dimensions
        if input_dims is None:
            input_dims = {mod: None for mod in modality_keys}
        else:
            assert set(modality_keys) == set(input_dims.keys()), "Input dimensions must be specified for all modalities."
        self.input_dims = input_dims
        
        # Set projection layers
        if unify_embeds:
            if out_proj_dim is None: # Determine output dimension
                assert all([dim is not None for dim in input_dims.values()]), "Either specify output dimension or input dimensions for all modalities."
                out_proj_dim = min([dim for dim in input_dims.values()])
             
            # Set projection dim  
            self.proj_dim = out_proj_dim
            
            if hidden_proj_dim is None: # Determine hidden dimension
                hidden_proj_dim = out_proj_dim
            
            self.projection_layers = nn.ModuleDict({
                mod: 
                    nn.Sequential(
                        ProjectionMLP(input_dim, hidden_proj_dim, out_proj_dim, bias, dropout) 
                        if input_dim is not None 
                        else LazyProjectionMLP(hidden_proj_dim, out_proj_dim, bias, dropout),
                        
                        L2NormalizationLayer(dim=-1) if normalize_proj else nn.Identity()
                    )
                for mod, input_dim 
                in self.input_dims.items()
            })            
        else:
            self.proj_dim = None
            
            self.projection_layers = nn.ModuleDict({
                mod: nn.Identity()
                for mod in modality_keys
            })
        
        
    def __call__(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(embeddings)
    
    @abstractmethod
    def _forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for the forward pass.
        """
        raise NotImplementedError("Forward method not implemented!")
        
    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Main forward pass.
        """
        # Project embeddings
        for mod in embeddings.keys():
            embed = embeddings[mod]
            
            if embed.dim() == 3: # Handle sequence data (B, T, E)                
                B, T, E = embed.shape
                embed = embed.reshape(B * T, E) # Flatten
                embed = self.projection_layers[mod](embed) # Project
                embed = embed.reshape(B, T, -1) # Reshape back
                
                embeddings[mod] = embed

            else: # (B, E)
                embeddings[mod] = self.projection_layers[mod](embed)
        
        # TODO: Consider automatic reweighting:
        # weight = nn.Parameter(torch.ones(1, self.out_proj_dim))
        # embed = embed * weight
        # for each modality
        
        # Perform fusion
        return self._forward(embeddings)