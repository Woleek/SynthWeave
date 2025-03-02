import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Classification loss functions for deep learning models.

This module provides various loss functions commonly used in classification tasks,
including standard cross-entropy losses and advanced metric learning losses that
enhance feature discrimination.
"""


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for multi-class classification.

    Combines LogSoftmax and NLLLoss in a single function. Useful for training
    models that output raw logits for multiple classes.

    Note:
        Expects unnormalized logits as input (before softmax).
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss.

        Args:
            logits: Raw model outputs of shape (batch_size, num_classes)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value
        """
        loss = F.cross_entropy(logits, labels)

        return loss


class BinaryCrossEntropyWithLogitsLoss(nn.Module):
    """Binary cross-entropy loss with logits for binary classification.

    Combines Sigmoid and BCELoss in a single function. More numerically stable
    than using a separate Sigmoid followed by BCELoss.

    Note:
        Expects unnormalized logits as input (before sigmoid).
    """

    def __init__(self):
        super(BinaryCrossEntropyWithLogitsLoss, self).__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute binary cross entropy loss with logits.

        Args:
            logits: Raw model outputs of shape (batch_size, 1)
            labels: Ground truth labels of shape (batch_size, 1)

        Returns:
            torch.Tensor: Scalar loss value
        """
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return loss


class CenterLoss(nn.Module):
    """Center loss for deep metric learning.

    Minimizes the distance between deep features and their corresponding class centers.
    Used alongside softmax loss to enhance intra-class compactness.

    Based on: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    Source: https://ydwen.github.io/papers/WenECCV16.pdf

    Attributes:
        num_classes: Number of classes in the dataset
        embedding_dim: Dimension of the feature embeddings
        lambda_c: Weight for the center loss term
        centers: Learnable class centers of shape (num_classes, embedding_dim)
    """

    def __init__(self, num_classes: int, embedding_dim: int, lambda_c: float = 1.0):
        """Initialize the center loss module.

        Args:
            num_classes: Number of classes in the dataset
            embedding_dim: Dimension of the feature embeddings
            lambda_c: Weight for the center loss term
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss.

        Args:
            embeddings: Feature embeddings of shape (batch_size, embedding_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value

        Process:
            1. Select centers corresponding to the input labels
            2. Compute squared distances between features and their centers
            3. Average the distances and apply lambda weight
        """
        batch_size = embeddings.size(0)
        centers_batch = self.centers[labels]
        loss = torch.sum((embeddings - centers_batch) ** 2) / (2.0 * batch_size)
        loss = self.lambda_c * loss
        return loss


class AMSoftmaxLoss(nn.Module):
    """Additive Margin Softmax loss for deep metric learning.

    Adds a margin to the target logit to enforce larger inter-class distances
    and smaller intra-class distances in the feature space.

    Based on: "Additive Margin Softmax for Face Verification"
    Source: https://arxiv.org/abs/1801.05599

    Attributes:
        embedding_dim: Dimension of the feature embeddings
        num_classes: Number of classes in the dataset
        margin: Additive margin to be applied
        scale: Scale factor for logits
        weight: Learnable class weights
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.35,
        scale: float = 30.0,
    ):
        """Initialize the AM-Softmax loss module.

        Args:
            embedding_dim: Dimension of the feature embeddings
            num_classes: Number of classes in the dataset
            margin: Additive margin to be applied
            scale: Scale factor for logits
        """
        super(AMSoftmaxLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute AM-Softmax loss.

        Args:
            embeddings: Feature embeddings of shape (batch_size, embedding_dim)
            labels: Ground truth labels of shape (batch_size,)

        Returns:
            torch.Tensor: Scalar loss value

        Process:
            1. Normalize embeddings and weights
            2. Compute cosine similarities
            3. Add margin to target logits
            4. Scale logits and compute cross entropy
        """
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)
        cosine = F.linear(embeddings, weight)
        target_logit = cosine - self.margin
        one_hot = torch.zeros_like(cosine)
        one_hot = one_hot.scatter(1, labels.view(-1, 1), 1.0)
        logits = one_hot * target_logit + (1.0 - one_hot) * cosine
        logits = logits * self.scale
        loss = F.cross_entropy(logits, labels)
        return loss


class AAMSoftmaxLoss(nn.Module):
    """
    (Additive Angular Margin Softmax)
    Variant of AM-Softmax that adds an angular margin for more robust embeddings.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
    ):
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

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        scale: float = 30.0,
        margin: float = 0.50,
    ):
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
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0.0, 1.0))

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
