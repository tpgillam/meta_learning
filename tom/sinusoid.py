import dataclasses
from typing import Tuple

import numpy
import torch

from tom.torch_extension import module_device


@dataclasses.dataclass(frozen=True)
class Batch:
    """Represent a batch of tasks, returned by `generate_sinusoid_batch`."""

    x: numpy.ndarray  # (batch_size_meta, batch_size_inner)
    y: numpy.ndarray  # (batch_size_meta, batch_size_inner)
    amplitude: numpy.ndarray  # (batch_size_meta,)
    phase: numpy.ndarray  # (batch_size_meta,)
    input_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]

    def __getitem__(self, slice_) -> 'Batch':
        """Slice over `batch_size_inner`."""
        return Batch(
            self.x[:, slice_],
            self.y[:, slice_],
            self.amplitude,
            self.phase,
            self.input_range,
            self.amplitude_range
        )

    def for_task(self, i_task: int) -> 'Batch':
        """Get a Batch for the given task index."""

        def slice_(arr):
            return arr[i_task:i_task + 1]

        return Batch(
            slice_(self.x),
            slice_(self.y),
            slice_(self.amplitude),
            slice_(self.phase),
            self.input_range,
            self.amplitude_range
        )


DEFAULT_INPUT_RANGE = (-5.0, 5.0)


def generate_sinusoid_batch(
        batch_size_meta: int,
        batch_size_inner: int,
        amplitude_range: Tuple[float, float] = (0.1, 5.0),
        phase_range: Tuple[float, float] = (0., numpy.pi),
        input_range: Tuple[float, float] = DEFAULT_INPUT_RANGE) -> Batch:
    """Compute a batch of samples.

    We draw `batch_size_meta` tasks, and for each task a batch of `batch_size_inner` points. Each "task"
    represents a regression problem, underlied by a sine wave with some amplitude and phase.

    Args:
        batch_size_meta: The number of tasks to draw.
        batch_size_inner: The number of samples for each task.
        amplitude_range: Draw the amplitude of the sine wave for the task uniformly from this range.
        phase_range: Draw the phase of the sine wave for the task uniformly from this range.
        input_range: The range from which the input variable will be drawn uniformly.
    """
    amplitude = numpy.random.uniform(amplitude_range[0], amplitude_range[1], batch_size_meta)
    phase = numpy.random.uniform(phase_range[0], phase_range[1], batch_size_meta)

    # All input locations are independent.
    x = numpy.random.uniform(
        input_range[0],
        input_range[1],
        (batch_size_meta, batch_size_inner))

    # To compute the outputs, we should broadcast the amplitude & phase over all inner samples.
    y = numpy.expand_dims(amplitude, axis=1) * numpy.sin(x - numpy.expand_dims(phase, axis=1))

    return Batch(x, y, amplitude, phase, input_range, amplitude_range)


def get_prediction_mse(module: torch.nn.Module, batch: Batch) -> torch.Tensor:
    """Get the MSE, as a tensor, for this module & batch."""
    device = module_device(module)
    x, y = batch_to_x_y(batch, device)

    # Forward pass: Compute predicted y by passing x to the model
    y_pred = module(x)

    # Compute loss.
    return torch.nn.functional.mse_loss(y_pred, y)


def batch_to_x_y(batch: Batch, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert the given batch to x & y torch tensors."""
    x = torch.tensor(batch.x.ravel(), dtype=torch.float32).unsqueeze(1).to(device)
    y = torch.tensor(batch.y.ravel(), dtype=torch.float32).unsqueeze(1).to(device)
    return x, y
