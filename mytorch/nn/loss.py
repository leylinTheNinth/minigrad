import numpy as np

import mytorch.nn.functional as F
from mytorch.tensor import Tensor


class Loss():
    """Base class for loss functions."""
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError("Loss subclasses must implement forward")


class CrossEntropyLoss(Loss):
    def __init__(self):
        pass

    def forward(self, predicted, target):
        # Simply calls nn.functional.cross_entropy
        return F.CrossEntropy.apply(predicted, target)
