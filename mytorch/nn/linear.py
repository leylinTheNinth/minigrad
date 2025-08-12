import numpy as np

from mytorch.nn.module import Module
from mytorch.tensor import Tensor


class Linear(Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Randomly initializing layer weights
        k = 1 / in_features
        weight = k * (np.random.rand(out_features, in_features) - 0.5)
        bias = k * (np.random.rand(out_features) - 0.5)
        self.weight = Tensor(weight, requires_grad=True, is_parameter=True)
        self.bias = Tensor(bias, requires_grad=True, is_parameter=True)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x @ self.weight.T + self.bias
