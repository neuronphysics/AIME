import torch

class Split(torch.nn.Module):
    """
    models a split in the network. works with convolutional models (not FC).
    specify out channels for the model to divide by n_parts.
    """
    def __init__(self, module, n_parts: int, dim=1):
        super().__init__()
        self._n_parts = n_parts
        self._dim = dim
        self._module = module

    def forward(self, inputs):
        output = self._module(inputs)
        chunk_size = output.shape[self._dim] // self._n_parts
        return torch.split(output, chunk_size, dim=self._dim)