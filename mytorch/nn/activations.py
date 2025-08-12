import mytorch.nn.functional as F
from mytorch.nn.module import Module


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):       
        return F.ReLU.apply(x)

