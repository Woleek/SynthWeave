import torch
import torch.nn as nn

from abc import abstractmethod
from typing import List, Optional

class BaseFusion(nn.Module):
    """
    Base class for all fusion modules.
    """
    def __init__(
            self, 
            output_dim: int,
            dropout: bool = True,
            input_dims: Optional[List[int]] = None,
        ):
        super(BaseFusion, self).__init__()
        
        # Store initialization parameters
        self._input_dims = input_dims
        self._output_dim = output_dim
        self._dropout = dropout
        
        # Register a buffer to track initialization state
        self.register_buffer('_fusion_initialized', torch.tensor(False), persistent=False)
        
    def __call__(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        return self.forward(embeddings)
    
    def _infer_input_dims(self, embeddings: List[torch.Tensor]) -> List[int]:
        """
        Infer input dimensions if not provided during initialization.
        """
        return [emb.shape[-1] for emb in embeddings]
    
    @abstractmethod
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for the forward pass.
        """
        raise NotImplementedError("Forward method not implemented!")
    
    @abstractmethod
    def _lazy_init(self):
        """
        Abstract method for lazy initialization of fusion-specific layers.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Fusion layer initialization not implemented!")
        
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with lazy initialization.
        """
        # Lazy initialization
        if not self._fusion_initialized:
            # Infer input dimensions if not provided
            if self._input_dims is None:
                self._input_dims = self._infer_input_dims(embeddings)
            
            # Perform fusion-specific initialization
            self._lazy_init()
            
            # Mark as initialized
            self._fusion_initialized.fill_(1)
        
        # Perform fusion
        return self._forward(embeddings)