import os
import pickle
from typing import List

import numpy
import torch
from contexttimer import Timer

from tom.learners import SinusoidalLearner
from tom.sinusoid import generate_sinusoid_batch, get_prediction_mse, batch_to_x_y


def anil_train(
        module: SinusoidalLearner,
        device: torch.device,
        *,
        num_steps_meta: int = 70000,
        num_steps_inner: int = 1,
        batch_size_meta: int = 25,
        batch_size_inner: int = 10,
        learning_rate_meta: float = 0.001,  # Meta Adam optimizer
        learning_rate_inner: float = 0.01,  # Inner SGD optimizer
        first_order: bool = False,  # Should we apply the first-order approximation?
        verbose: bool = False) -> List[float]:
    module.to(device)

    if verbose:
        print(module)

    # Construct an Optimizer. The call to model.parameters() in the constructor will contain the
    # learnable parameters of the module.
    meta_optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate_meta)

    # These are the MSEs from the 'test' part of the batches drawn within the meta-training process.
    meta_train_test_mses = []

    for i_meta in range(num_steps_meta):

        # Generate a batch with which to train.
        batch = generate_sinusoid_batch(batch_size_meta, batch_size_inner * 2)

        train_batch = batch[:batch_size_inner]
        test_batch = batch[batch_size_inner:]

        total_losses = []
        for i_task in range(batch_size_meta):
            # 0. Get task-specific batches
            task_train_batch = train_batch.for_task(i_task)
            task_test_batch = test_batch.for_task(i_task)

            # 1. Forward on the original module
            loss = get_prediction_mse(module, task_train_batch)

            # 2. Use gradients to build *new* parameters
            # create_graph=True is required for allowing higher-order derivatives. Note that we
            # get the wrong answer below if we set it to False! NB - we *may* in practice want to experiment
            # with setting it to False, though.
            # Effectively this would drop second-order terms when we come to perform the meta-training step.
            # ANIL: only modify the last layer.
            inner_parameters = list(module.layers[-1].parameters())

            d_loss_d_parameters = torch.autograd.grad(
                [loss],
                inner_parameters,
                create_graph=not first_order)

            parameter_to_replacement = {
                parameter: parameter - learning_rate_inner * d_loss_d_parameter
                for parameter, d_loss_d_parameter in zip(inner_parameters, d_loss_d_parameters)
            }

            if num_steps_inner != 1:
                raise NotImplementedError

            # 3. Forward with new parameters on test batch to build loss used in meta-train step.
            x, y = batch_to_x_y(task_test_batch, device)
            y_pred = module.forward_with_parameter_replacement(x, parameter_to_replacement)
            total_losses.append(torch.nn.functional.mse_loss(y_pred, y))

        # Whilst the paper uses a sum, in practice the reference implementation uses a mean.
        total_loss = sum(total_losses) / batch_size_meta

        # Apply the meta-optimisation step to the module parameters.
        meta_optimizer.zero_grad()
        total_loss.backward()
        meta_optimizer.step()

        if i_meta % 100 == 0:
            mse_loss = total_loss.item()
            meta_train_test_mses.append(mse_loss)
            if verbose:
                print(f'Epoch {i_meta + 1}/{num_steps_meta}:   {mse_loss}')

    return meta_train_test_mses


def main():
    # TODO Make these command-line arguments
    device = torch.device('cuda:0')
    experiment_name = "70k"
    first_order = False
    verbose = True

    # Aim for reproducibility.
    numpy.random.seed(42)
    torch.manual_seed(42)

    module = SinusoidalLearner()

    with Timer() as timer:
        mses = anil_train(module, device, first_order=first_order, verbose=verbose)
    print(f"Time: {timer}")

    # Save the output to disk.
    with open(os.path.join('trained_models', f'sinusoid_anil_{experiment_name}_train_mse.pkl'), 'wb') as file_:
        pickle.dump(mses, file_)
    torch.save(module.state_dict(), f'trained_models/sinusoid_anil_{experiment_name}.pt')


if __name__ == '__main__':
    main()
