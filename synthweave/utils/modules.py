from torch import nn


class LazyLinearXavier(nn.LazyLinear):
    def __init__(self, out_features: int):
        super(LazyLinearXavier, self).__init__(out_features)

    def reset_parameters(self):
        if not self.has_uninitialized_params() and self.in_features != 0:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.constant_(self.bias, 0.01)
                
class LinearXavier(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super(LinearXavier, self).__init__(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.01)