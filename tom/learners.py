from typing import Collection, Mapping

import numpy
import torch
from more_itertools import pairwise
from scipy.stats import truncnorm


class SinusoidalLearner(torch.nn.Module):
    """A module suitable for mimicking a sine wave with some unknown amplitude and phase."""

    def __init__(self, hidden_sizes: Collection[int] = (40, 40)):
        super().__init__()
        dim_in = 1
        dim_out = 1
        all_dims = (dim_in,) + tuple(hidden_sizes) + (dim_out,)
        self.layers = [torch.nn.Linear(dim_1, dim_2) for dim_1, dim_2 in pairwise(all_dims)]

        for i, layer in enumerate(self.layers):
            # In the reference implementation, initialisation is:
            #    * zero for bias terms
            #    * truncated_normal, std=0.01 for weights.
            torch.nn.init.zeros_(layer.bias)

            # Truncated normal not available in raw pytorch, so compute the values with scipy and
            # copy into memory
            _assign_to_tensor(
                layer.weight,
                truncnorm.rvs(-2, 2, scale=0.01, size=layer.weight.shape))

            # Register this as a sub-module, so we declare the existence of the necessary parameters
            self.add_module(f'layer_{i}', layer)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # We should not apply the non-linearity on the last layer!
            is_last = i == len(self.layers) - 1
            if not is_last:
                x = torch.nn.functional.relu(x)
        return x

    def forward_with_parameter_replacement(
            self,
            x: torch.Tensor,
            parameter_to_replacement: Mapping[torch.nn.Parameter, torch.Tensor]):
        """Equivalent to `forward`, but using custom parameter tensors instead."""
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, torch.nn.Linear)

            weight = parameter_to_replacement.get(layer.weight, layer.weight)
            bias = parameter_to_replacement.get(layer.bias, layer.bias)

            x = torch.nn.functional.linear(x, weight, bias)

            # We should not apply the non-linearity on the last layer!
            is_last = i == len(self.layers) - 1
            if not is_last:
                x = torch.nn.functional.relu(x)
        return x


def _assign_to_tensor(tensor: torch.Tensor, value: numpy.ndarray):
    """Assign the contents of `value` to tensor."""
    with torch.no_grad():
        tensor[:] = torch.from_numpy(value)