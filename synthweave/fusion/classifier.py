import torch
import torch.nn as nn
from typing import List, Optional

from .. import logger
from .base import BaseFusion
from ..utils.modules import LazyLinearXavier

class MV(BaseFusion):
    """
    Majority voting (MV) module.
    
    Baseline method, where the final prediction is based on the majority vote of the predictions from each modality.
    
    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103
    """
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True
    ) -> None:
        """
        Initializes the MV module.
        """
        super(MV, self).__init__(dropout, unify_embeds)
        
        logger.warning("Note that this method outputs predictions instead of vectors and does not require additional classifier head.")
        
        if len(n_modals) % 2 == 0:
            raise ValueError("Majority voting requires an odd number of modalities.")
        
        if output_dim != 1:
            logger.warning("This method is designed for binary classification. Setting `output_dim` to 1.")
            output_dim = 1
            
        # Unify representations into same dimension
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList([
                LazyLinearXavier(output_dim)
                for _ in range(n_modals)
            ])
        else:
            self.unify_layers = nn.ModuleList([
                nn.Identity()
                for _ in range(n_modals)
            ])
        
        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList([
            nn.Sequential(
                LazyLinearXavier(output_dim),
                nn.Sigmoid()
            )
            for _ in range(n_modals)
        ])
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the MV module.
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
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
    """
    Average Score Fusion (ASF) module.
    
    Baseline method, where the final prediction is based on the average score of the predictions from each modality.
    
    Based on: "AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection"
    Source: https://arxiv.org/abs/2310.13103
    """
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True
    ) -> None:
        """
        Initializes the ASF module.
        """
        super(ASF, self).__init__(dropout, unify_embeds)
        
        logger.warning("Note that this method outputs predictions instead of vectors and does not require additional classifier head.")
        
        if output_dim != 1:
            logger.warning("This method is designed for binary classification. Setting `output_dim` to 1.")
            output_dim = 1
            
        # Unify representations into same dimension
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList([
                LazyLinearXavier(output_dim)
                for _ in range(n_modals)
            ])
        else:
            self.unify_layers = nn.ModuleList([
                nn.Identity()
                for _ in range(n_modals)
            ])
        
        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList([
            nn.Sequential(
                LazyLinearXavier(output_dim),
                nn.Sigmoid()
            )
            for _ in range(n_modals)
        ])
        
        # Threshold for binarization
        self.threshold = 0.5
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the ASF module.
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]
        
        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, proj_embeds)]
        
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
    def __init__(
        self, 
        output_dim: int,
        n_modals: int,
        dropout: bool = True,
        unify_embeds: bool = True
    ) -> None:
        """
        Initializes the SF module.
        """
        super(SF, self).__init__(dropout, unify_embeds)
        
        logger.warning("Note that this method outputs predictions instead of vectors and does not require additional classifier head.")
        
        if output_dim != 1:
            logger.warning("This method is designed for binary classification. Setting `output_dim` to 1.")
            output_dim = 1
            
        # Unify representations into same dimension
        if self._unify_embeds:
            self.unify_layers = nn.ModuleList([
                LazyLinearXavier(output_dim)
                for _ in range(n_modals)
            ])
        else:
            self.unify_layers = nn.ModuleList([
                nn.Identity()
                for _ in range(n_modals)
            ])
        
        # Linear classifiers for each modality
        self.clf_layers = nn.ModuleList([
            LazyLinearXavier(output_dim)
            for _ in range(n_modals)
        ])
        
        self.score_fusion_layer = nn.Sequential(
            LazyLinearXavier(output_dim),
            nn.Sigmoid()
        )
        
    def _forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the SF module.
        """
        # Unify representations into same dimension
        proj_embeds = [
            unify_layer(embed) 
            for unify_layer, embed 
            in zip(self.unify_layers, embeddings)
        ]
        
        # Get probabilities from each modality
        logits = [layer(embed) for layer, embed in zip(self.clf_layers, proj_embeds)]
        
        # Score fusion
        scores = torch.stack(logits, dim=-1)
        preds = self.score_fusion_layer(scores)
        
        # Binarize predictions
        results = torch.round(preds)
        
        return results.view(-1, 1)