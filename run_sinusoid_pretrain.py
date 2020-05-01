import os
import pickle
from typing import NamedTuple, List

import numpy
import torch
from contexttimer import Timer

from tom.learners import SinusoidalLearner
from tom.sinusoid import generate_sinusoid_batch, Batch, get_prediction_mse, batch_to_x_y
from tom.torch_extension import module_eval, module_device


def _train_step(module: torch.nn.Module, train_batch: Batch, optimizer):
    loss = get_prediction_mse(module, train_batch)

    # Zero gradients, perform a backward pass (compute gradients wrt. the loss), and update the
    # weights accordingly.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def _get_prediction(module: torch.nn.Module, x: numpy.ndarray) -> numpy.ndarray:
    device = module_device(module)
    torch_x = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad(), module_eval(module):
        torch_y = module(torch_x)
    return torch_y.cpu().numpy().squeeze(1)


class BatchPrediction(NamedTuple):
    y: numpy.ndarray
    mse_loss: float


def _get_test_prediction(module, test_batch: Batch) -> BatchPrediction:
    """Get the prediction when evaluating the module on the given batch."""
    device = module_device(module)
    x, y = batch_to_x_y(test_batch, device)

    # No need to accumulate gradients when evaluating the validation loss.
    # Also, we put the model into "evaluation mode" for the purpose of computing the prediction.
    # This is to prevent layers like BatchNorm / Dropout mutating their internal state.
    with torch.no_grad(), module_eval(module):
        y_pred = module(x)
        test_mse = torch.nn.functional.mse_loss(y_pred, y).item()

    return BatchPrediction(y_pred.cpu().numpy().squeeze(1), test_mse)


def pretrain(
        module: SinusoidalLearner,
        device: torch.device,
        *,
        num_steps_train: int = 70000,
        batch_size_meta: int = 25,
        batch_size_inner: int = 10,
        learning_rate: float = 0.001,
        verbose: bool = False) -> List[float]:
    """Perform pre-training.

    :param module: The module to train - will be mutated by this function
    :param device: Torch device to use for training.
    :param num_steps_train: The number of pre-training steps to perform.
    :param batch_size_meta: The number of tasks in each meta-batch.
    :param batch_size_inner: The size of the training set. We actually generate *twice* this amount of data in total;
     half is used for the inner training loop, and the other half for inner testing.
    :param learning_rate: Learning rate used for the meta-optimiser.
    :param verbose: Whether to perform additional printing.

    :return: The mean-squared-errors from training.
    """
    module = module.to(device)

    if verbose:
        print(module)

    # Construct an Optimizer. The call to model.parameters() in the constructor will contain the
    # learnable parameters of the module.
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)

    # These are the MSEs from the 'test' part of the batches drawn within the meta-training process.
    meta_train_test_mses = []

    for i_meta in range(num_steps_train):

        # Generate a batch with which to train.
        batch = generate_sinusoid_batch(batch_size_meta, batch_size_inner * 2)

        train_batch = batch[:batch_size_inner]
        test_batch = batch[batch_size_inner:]

        _train_step(module, train_batch, optimizer)

        if i_meta % 100 == 0:
            prediction = _get_test_prediction(module, test_batch)
            meta_train_test_mses.append(prediction.mse_loss)
            if verbose:
                print(f'Epoch {i_meta + 1}/{num_steps_train}:   {prediction.mse_loss}')

    return meta_train_test_mses


def main():
    # TODO Make this a command-line argument
    device = torch.device('cuda:0')
    verbose = True
    experiment_name = '70k'

    # Aim for reproducibility.
    numpy.random.seed(42)
    torch.manual_seed(42)

    module = SinusoidalLearner()

    with Timer() as timer:
        mses = pretrain(module, device, verbose=verbose)
    print(f"Time: {timer}")

    # Save the output to disk.
    with open(os.path.join('trained_models', f'sinusoid_pretrain_{experiment_name}_train_mse.pkl'), 'wb') as file_:
        pickle.dump(mses, file_)
    torch.save(module.state_dict(), os.path.join('trained_models', f'sinusoid_pretrain_{experiment_name}.pt'))


if __name__ == '__main__':
    main()
