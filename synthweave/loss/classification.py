import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Standard loss function for classification tasks.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
class BinaryCrossEntropyWithLogitsLoss(nn.Module):
    """
    Standard loss function for binary classification tasks.
    """
    def __init__(self):
        super(BinaryCrossEntropyWithLogitsLoss, self).__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        return loss
    
class CenterLoss(nn.Module):
    """
    Minimizes intra-class variance in the feature space while maintaining inter-class separability. Often used alongside CrossEntropy Loss.
    """
    def __init__(self, num_classes: int, embedding_dim: int, lambda_c: float = 1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Compute the distances between features and their class centers
        batch_size = embeddings.size(0)
        
        # Select corresponding centers
        centers_batch = self.centers[labels]  
        
        # Compute loss
        loss = torch.sum((embeddings - centers_batch) ** 2) / (2.0 * batch_size)
        loss = self.lambda_c * loss 
        
        return loss
    
class AMSoftmaxLoss(nn.Module):
    """
    Introduces an additive margin to the softmax function to enhance inter-class separability.
    """
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.35, scale: float = 30.0):
        super(AMSoftmaxLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and input
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        target_logit = cosine - self.margin

        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot = one_hot.scatter(1, labels.view(-1, 1), 1.0)

        # Apply margin
        logits = one_hot * target_logit + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
class AAMSoftmaxLoss(nn.Module):
    """
    (Additive Angular Margin Softmax)
    Variant of AM-Softmax that adds an angular margin for more robust embeddings.
    """
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.2, scale: float = 30.0):
        super(AAMSoftmaxLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and input
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(torch.clamp(cosine, -1.0, 1.0))
        target_logit = torch.cos(theta + self.margin)

        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot = one_hot.scatter(1, labels.view(-1, 1), 1.0)
        
        # Apply margin
        logits = one_hot * target_logit + (1.0 - one_hot) * cosine
        logits = logits * self.scale
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)

        return loss
    
class ArcMarginLoss(nn.Module):
    """
    Introduces an angular margin to the softmax function to enhance inter-class separability.
    """
    def __init__(self, num_classes: int, embedding_dim: int, scale: float=30.0, margin: float=0.50):
        super(ArcMarginLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and input
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0.0, 1.0))
        
        # Compute theta + margin
        phi = cosine * torch.cos(self.margin) - sine * torch.sin(self.margin)
        
        # Create one-hot encoding for labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        
        # Apply margin
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = self.scale * logits
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss