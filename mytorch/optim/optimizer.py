import numpy as np


class Optimizer():
    """Base class for optimizers"""
    def __init__(self, params):
        self.params = list(params)
        self.state = [] # Technically supposed to be a dict in real torch

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad = None
