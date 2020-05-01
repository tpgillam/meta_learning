import contextlib

import torch


@contextlib.contextmanager
def module_eval(module: torch.nn.Module):
    """Enter 'evaluation mode' for the given module. """
    module.eval()  # DANGER: This *mutates* the module. It returns a reference to itself
    try:
        yield
    finally:
        module.train()


def module_device(module: torch.nn.Module) -> torch.device:
    """Find the PyTorch device to which the given module is bound."""
    devices = {x.device for x in module.parameters()}
    if len(devices) != 1:
        raise ValueError(f"Found candidate devices {devices}")
    return next(iter(devices))
