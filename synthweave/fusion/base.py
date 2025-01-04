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
            dropout: bool = True,
            unify_embeds: bool = True
        ):
        super(BaseFusion, self).__init__()
        
        # Store initialization parameters
        self._dropout = dropout
        self._unify_embeds = unify_embeds  
        
    def __call__(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        return self.forward(embeddings)
    
    @abstractmethod
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for the forward pass.
        """
        raise NotImplementedError("Forward method not implemented!")
        
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Main forward pass.
        """
        #TODO: Check if L2 normalization is required
        # embeddings = [nn.functional.normalize(embed, p=2, dim=1) for embed in embeddings]
        
        # Perform fusion
        return self._forward(embeddings)