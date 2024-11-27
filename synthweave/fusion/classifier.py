import torch
import torch.nn as nn
from typing import List, Optional

from .. import logger
from .base import BaseFusion

class MV(BaseFusion):
    """
    Majority voting (MV) module.
    
    Baseline method, where the final prediction is based on the majority vote of the predictions from each modality.
    
    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103
    """
    def __init__(self, output_dim: int = 1, dropout: bool = True, input_dims: Optional[List[int]] = None) -> None:
        super(MV, self).__init__(output_dim, dropout, input_dims)
        logger.warning("Note that this method outputs predictions instead of vectors and does not require additional classifier head.")
        
        if output_dim != 1:
            logger.warning("This method is designed for binary classification. Setting `output_dim` to 1.")
            self._output_dim = 1
    
    def _lazy_init(self) -> None:
        """
        Initializes the MV module.
        """
        
        if len(self._input_dims) % 2 == 0:
            raise ValueError("Majority voting requires an odd number of modalities.")
        
        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self._output_dim),
                nn.Sigmoid()
            )
            for input_dim
            in self._input_dims
        ])
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the MV module.
        """
        
        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, embeddings)]
        
        # Binarize predictions
        preds = [torch.round(logit) for logit in logits]
        
        # Majority voting
        votes = torch.stack(preds, dim=-1)
        results = torch.mode(votes, dim=-1).values.float()
        
        return results.view(-1, 1)
        
class ASF(BaseFusion):
    """
    Average Score Fusion (ASF) module.
    
    Baseline method, where the final prediction is based on the average score of the predictions from each modality.
    
    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103
    """
    def __init__(self, output_dim: int = 1, dropout: bool = True, input_dims: Optional[List[int]] = None) -> None:
        super(ASF, self).__init__(output_dim, dropout, input_dims)
        logger.warning("Note that this method outputs predictions instead of vectors and does not require additional classifier head.")
        
        if output_dim != 1:
            logger.warning("This method is designed for binary classification. Setting `output_dim` to 1.")
            self._output_dim = 1
    
    def _lazy_init(self) -> None:
        """
        Initializes the ASF module.
        """
        
        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self._output_dim),
                nn.Sigmoid()
            )
            for input_dim
            in self._input_dims
        ])
        
        # Threshold for binarization
        self.threshold = 0.5
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the ASF module.
        """
        
        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, embeddings)]
        
        # Average scores
        avg_scores = torch.stack(logits, dim=-1).mean(dim=-1)
        
        # Binarize predictions
        results = (avg_scores > self.threshold).float()
        
        return results.view(-1, 1)
    
class SF(BaseFusion):
    """
    Score Fusion (SF) module.
    
    Baseline method, where the final prediction is based on the score of the predictions from each modality.
    
    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103
    """
    def __init__(self, output_dim: int = 1, dropout: bool = True, input_dims: Optional[List[int]] = None) -> None:
        super(SF, self).__init__(output_dim, dropout, input_dims)
        logger.warning("Note that this method outputs predictions instead of vectors and does not require additional classifier head.")
        
        if output_dim != 1:
            logger.warning("This method is designed for binary classification. Setting `output_dim` to 1.")
            self._output_dim = 1
    
    def _lazy_init(self) -> None:
        """
        Initializes the SF module.
        """
        
        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList([
            nn.Linear(input_dim, self._output_dim)
            for input_dim
            in self._input_dims
        ])
        
        self.score_fusion_layer = nn.Sequential(
            nn.Linear(len(self._input_dims), self._output_dim),
            nn.Sigmoid()
        )
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the SF module.
        """
        
        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, embeddings)]
        
        # Score fusion
        scores = torch.stack(logits, dim=-1)
        preds = self.score_fusion_layer(scores)
        
        # Binarize predictions
        results = torch.round(preds)
        
        return results.view(-1, 1)