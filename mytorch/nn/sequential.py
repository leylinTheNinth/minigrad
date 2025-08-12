from mytorch.nn.module import Module

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

        # iterate through args provided and store them
        for idx, l in enumerate(self.layers):
            self.add_module(str(idx), l)

    def __iter__(self):
        yield from self.layers

    def __getitem__(self, idx):
        """Enables list-like indexing for layers"""
        return self.layers[idx]

    def train(self):
        """Sets this object and all trainable modules within to train mode"""
        self.is_train = True
        for submodule in self._submodules.values():
            submodule.train()

    def eval(self):
        self.is_train = False
        for submodule in self._submodules.values():
            submodule.eval()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        

