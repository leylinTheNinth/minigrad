import numpy as np

import mytorch.autograd_engine as autograd_engine
from mytorch.nn import functional as F
from mytorch.autograd_engine import AccumulateGrad


class Tensor:
    """Tensor object, similar to `torch.Tensor`
    A wrapper around a NumPy array that help it interact with MyTorch.
    """
    def __init__(self, data, requires_grad=False, is_leaf=True,
                 is_parameter=False):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.grad_fn = None # Set during forward pass
        self.grad = None
        self.is_parameter = is_parameter

    def __str__(self):
        return "{}{}".format(
            str(self.data),
            ", grad_fn={}".format(self.grad_fn.__class__.__name__) if self.grad_fn is not None else ""
        )

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self):
        return self.data.shape

    def fill_(self, fill_value):
        """In-place operation, replaces data with repeated value"""
        self.data.fill(fill_value)
        return self

    def copy(self):
        return Tensor(self.data)

    # Below methods can be used WITHOUT creating a tensor first
    # (For example, we can call Tensor.zeros(3,2) directly)

    @staticmethod
    def zeros(*shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def ones(*shape):
        return Tensor(np.ones(shape))

    @staticmethod
    def arange(*interval):
        return Tensor(np.arange(*interval))

    @staticmethod
    def randn(*shape):
        return Tensor(np.random.normal(0, 1, shape))

    @staticmethod
    def empty(*shape):
        return Tensor(np.empty(shape))

    def backward(self):
        if not self.requires_grad:
            raise Exception("Cannot call backward on Tensor which doesn't require gradient")
        grad_output = Tensor(np.ones_like(self.data))

        if self.grad_fn is not None:
            autograd_engine.backward(self.grad_fn, grad_output)

    @property
    def T(self):
        return F.Transpose.apply(self)

    def reshape(self, *shape):
        return F.Reshape.apply(self, shape)

    def log(self):
        return F.Log.apply(self)

    def __add__(self, other):
        return F.Add.apply(self, other)

    def __sub__(self, other):
        return F.Sub.apply(self, other)
    
    def __mul__(self, other):
        return F.Mul.apply(self, other)
    
    def __truediv__(self, other):
        return F.Div.apply(self, other)
    
    def sum(self, axis=None, keepdims=False):
        return F.Sum.apply(self, axis, keepdims)
    
    def __matmul__(self, other):
        return F.MatMul.apply(self, other)

    # TODO: have to implement more functions below

