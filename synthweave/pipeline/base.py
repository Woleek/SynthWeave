import torch
import torch.nn as nn

from ..fusion.base import BaseFusion
from typing import Any, Dict, Optional, Callable, Union, Mapping, List

class BasePipeline(nn.Module):
    """
    Base class implementing pipelined operations. Pipeline workflow is defined as a sequence of the following
    operations:

        Uni-modal inputs -> Preprocessing -> Feature Extraction -> Fusion -> Head (task specific) -> Output
    """
    def __init__(
            self, 
            models: Mapping[str, nn.Module], 
            fusion: Union[BaseFusion, str], 
            head: Optional[nn.Module] = None,
            processors: Optional[Mapping[str, Callable[..., torch.Tensor]]] = None,
            freeze_backbone: bool = True,
        ):
        super(BasePipeline, self).__init__()

        self.feature_extractors = nn.ModuleDict(models)
        if freeze_backbone:
            self._freeze_backbone()
        self.fusion = fusion
        self.head = head
        self.processors = processors
        
    def __call__(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return self.forward(inputs)
    
    # def to(self, device: torch.device) -> 'BasePipeline':
    #     """
    #     Override the to method to move all models to the new device.
    #     """
    #     self.feature_extractors = self.feature_extractors.to(device)
    #     self.fusion = self.fusion.to(device)
    #     if self.head is not None:
    #         self.head = self.head.to(device)
    #     return super(BasePipeline, self).to(device)
    
    def _freeze_backbone(self) -> None:
        """
        Freeze the backbone models.
        """
        for model in self.feature_extractors.values():
            for param in model.parameters():
                param.requires_grad = False
    
    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess the inputs.
        """
        if self.processors is not None:
            for modality, processor in self.processors.items():
                inputs[modality] = processor(inputs[modality])
        return inputs
    
    def extract_features(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Extract features from each modality.
        """
        embeddings = []
        for modality, model in self.feature_extractors.items():
            embeddings.append(model(inputs[modality]))
        return embeddings
        
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Run the pipeline.
        """
        # Preprocess inputs
        inputs = self.preprocess(inputs)
        
        # Extract embeddings from each modality
        embeddings = self.extract_features(inputs)
        
        # Fuse embeddings into one vector
        fused_embedding = self.fusion(embeddings)
        
        # Pass the fused embedding to the head
        if self.head is not None:
            output = self.head(fused_embedding)
        else:
            output = fused_embedding
        
        return output