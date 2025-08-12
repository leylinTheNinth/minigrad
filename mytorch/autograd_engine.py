from mytorch import tensor

def backward(grad_fn, grad_of_outputs):
    parent_grads = grad_fn.apply(grad_of_outputs)

    if not isinstance(parent_grads, tuple):
        parent_grads = (parent_grads,)

    for i, next_fn in enumerate(grad_fn.next_functions):
        if next_fn is not None and i < len(parent_grads):
            backward(next_fn, parent_grads[i])


class Function:
    """Superclass for linking nodes to the computational graph.
    Operations in `functional.py` should inherit from this"""
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("All subclasses must implement forward")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("All subclasses must implement backward")

    @classmethod
    def apply(cls, *args):
        backward_function = BackwardFunction(cls)
        output_tensor = cls.forward(backward_function.ctx, *args)
        
        for arg in args:
            if isinstance(arg, tensor.Tensor):  # Make sure it's a tensor
                node_to_append = None  # Default for constant nodes
                
                if arg.requires_grad:
                    if arg.is_leaf and arg.grad_fn is None:
                        # Case 1: AccumulateGrad node (leaf that requires grad)
                        arg.grad_fn = AccumulateGrad(arg)  # Use the tensor itself
                    # Use existing grad_fn (either AccumulateGrad we just created or BackwardFunction)
                    node_to_append = arg.grad_fn
                
                # Append the appropriate node (or None for constants)
                backward_function.next_functions.append(node_to_append)

        output_tensor.grad_fn = backward_function

        return output_tensor


class AccumulateGrad:
    """Represents node where gradient must be accumulated.
    Args:
        tensor (Tensor): The tensor where the gradients are accumulated in `.grad`
    """
    def __init__(self, tensor):
        self.variable = tensor
        self.next_functions = [] # nodes of current node's parents (this WILL be empty)
                                 # exists just to be consistent in format with BackwardFunction
        self.function_name = "AccumulateGrad" # just for convenience lol

    def apply(self, arg):
        # if no grad stored yet, initialize. otherwise +=
        if self.variable.grad is None:
            self.variable.grad = tensor.Tensor(arg.data)
        else:
            self.variable.grad.data += arg.data

        shape = self.variable.shape
        grad_shape = self.variable.grad.shape
        assert shape == grad_shape, (shape, grad_shape)

class ContextManager:
    def __init__(self):
        self.saved_tensors = [] # list that TENSORS get stored in

    def save_for_backward(self, *args):
        for arg in args:
            # Raises error if arg is not tensor (i warned you)
            if type(arg).__name__ != "Tensor":
                raise Exception("Got type {} of object {}. \nOnly Tensors should be saved in save_for_backward. For saving constants, just save directly as a new attribute.".format(type(arg), arg))

            self.saved_tensors.append(arg.copy())


class BackwardFunction:
    def __init__(self, cls):
        self.ctx = ContextManager() # Just in case args need to be passed (see above)
        self._forward_cls = cls

        # Nodes of parents, populated in `Function.apply`
        self.next_functions = []

        # The name of the operation as a string (for convenience)
        self.function_name = cls.__name__

    def apply(self, *args):
        """Generates gradient by running the operation's `.backward()`.
        Args:
            args: Args for the operation's `.backward()`
        Returns:
            Tensor: gradient of parent's output w.r.t. current output
        """
        # Note that we've already provided the ContextManager
        return self._forward_cls.backward(self.ctx, *args)
