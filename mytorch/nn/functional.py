import numpy as np

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)
    

class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for ReLU must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.maximum(0, a.data), requires_grad=requires_grad, is_leaf= not requires_grad)
        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0] # it should be only one but for safety
        mask = (a.data > 0).astype(float)
        return tensor.Tensor(grad_output.data * mask)

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        if a.data.shape != b.data.shape:
            raise Exception("Args must be of same shape: {}, {}".format(a.data.shape, b.data.shape))

        # saving input to access in backward pass
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad= requires_grad, is_leaf= not requires_grad)

        return c


    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = -1*np.ones(b.shape) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None

# this is element wise multiplication in tensors
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both Args must be tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        # shape does not have to be equal as numpy can broadcast 
        
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(a.data*b.data, requires_grad=requires_grad, is_leaf= not requires_grad)

        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # dL/da = dOut/da * dL/dOut
        grad_a = b.data * grad_output.data 
        grad_b = a.data * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

# this is element wise division
class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both Args must be tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        ctx.save_for_backward(a, b)

        if np.any(b.data == 0):
            raise Exception("Division by zero encountered in tensor division")
        
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(a.data/b.data, requires_grad=requires_grad, is_leaf = not requires_grad)

        return c
    
    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        #dL/da = 1/b * grad_output.data
        grad_a = grad_output.data/b.data
        #dL/db = -a/b*b * grad_output.data
        grad_b = -1* a.data/(b.data*b.data) * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b

# this is matrix multiplication
class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both Args must be tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data@b.data, requires_grad=requires_grad, is_leaf= not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors 
        # Y = XA
        #dL/dx = dL/dY * dY/dx
        grad_x = grad_output.data @ a.data.T
        #dL/dA = dL/dY * dY/dA
        grad_a = x.data.T @ grad_output.data

        grad_x = tensor.Tensor(grad_x)
        grad_a = tensor.Tensor(grad_a)

        return grad_x, grad_a
        
# cross entropy operation (includes softmax)
class CrossEntropy(Function):
    @staticmethod
    def forward(ctx, predicted, target): 
        if not (type(predicted).__name__ == 'Tensor' and type(target).__name__ == "Tensor"):
            raise Exception("Both Args must be tensors: {}, {}".format(type(predicted).__name__, type(target).__name__)) 
        
        batch_size, num_classes = predicted.shape 

        logits = predicted.data - np.max(predicted.data, keepdims= True, axis= 1)
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits/np.sum(exp_logits, axis= 1, keepdims= True)

        if len(target.shape) == 1:
            target_one_hot = to_one_hot(target, num_classes).data
        else:
            target_one_hot = target.data
        
        #let's save for backward calculation
        ctx.softmax_probs = softmax_probs
        ctx.batch_size = batch_size
        ctx.target_one_hot = target_one_hot

        loss = -np.sum(target_one_hot * np.log(softmax_probs + 1e-10))/batch_size
        requires_grad = predicted.requires_grad
        
        return tensor.Tensor(loss, requires_grad=requires_grad, is_leaf= not requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax_probs = ctx.softmax_probs
        target_one_hot = ctx.target_one_hot
        batch_size = ctx.batch_size
        
        # gradient wrt to logits
        grad_logits  = (softmax_probs - target_one_hot)/batch_size

        grad_logits = grad_logits * grad_output.data

        return tensor.Tensor(grad_logits), None

def to_one_hot(arr, num_classes):
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = False)

