from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class LazyLinearXavier(nn.LazyLinear):
    def __init__(self, out_features: int):
        super(LazyLinearXavier, self).__init__(out_features)

    def reset_parameters(self):
        if not self.has_uninitialized_params() and self.in_features != 0:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.01)

class LinearXavier(nn.Linear):
    """A linear layer with Xavier uniform initialization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
            bias (bool, optional): If True, adds a learnable bias. Defaults to True.
        """
        super(LinearXavier, self).__init__(in_features, out_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the layer parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.01)


class ProjectionMLP(nn.Module):
    """A projection MLP with two linear layers, batch normalization, ReLU, and dropout.

    The architecture follows: Linear -> BatchNorm -> ReLU -> Dropout -> Linear
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool = True,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension
            bias (bool, optional): If True, adds learnable bias to linear layers. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
        """
        super(ProjectionMLP, self).__init__()

        self.proj = nn.Sequential(
            OrderedDict([
                    ("linear1", nn.Linear(in_dim, in_dim)),
                    ("bn1",   nn.BatchNorm1d(in_dim)),
                    ("relu1",   nn.LeakyReLU()),
                    ("drop1",   nn.Dropout(dropout)),

                    ("linear2", nn.Linear(in_dim, out_dim)),
                    # ("relu2",   nn.LeakyReLU()),
                    # ("drop2",   nn.Dropout(dropout)),

                    # ("linear3", nn.Linear(out_dim, out_dim)),
                ]
            )
        )

    def forward(self, x):
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.proj(x)


class LazyProjectionMLP(nn.Module):
    """A lazy projection MLP with two linear layers, batch normalization, ReLU, and dropout.

    Similar to ProjectionMLP but with a lazy first layer that infers input dimension.
    """

    def __init__(
        self, hidden_dim: int, out_dim: int, bias: bool = True, dropout: float = 0.5
    ):
        """
        Args:
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output dimension
            bias (bool, optional): If True, adds learnable bias to linear layers. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
        """
        super(LazyProjectionMLP, self).__init__()

        self.proj = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.LazyLinear(hidden_dim, bias)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(dropout)),
                    ("linear2", nn.Linear(hidden_dim, out_dim, bias)),
                ]
            )
        )

    def forward(self, x):
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.proj(x)


class L2NormalizationLayer(nn.Module):
    """A layer that performs L2 normalization on the input tensor."""

    def __init__(self, dim=1, eps=1e-12):
        """
        Args:
            dim (int, optional): Dimension along which to normalize. Defaults to -1.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-12.
        """
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        """Forward pass performing L2 normalization.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: L2 normalized tensor
        """
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
