"""Contains the train and test functions used to iterate the NN model in the main
module.
"""
import logging
from typing import Callable

import torch

import theoretical_nn_training.generators as generators


def train(
    model: torch.nn.Module,
    generator: generators.Generator,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_batches: int,
    avg_loss: torch.Tensor,
) -> torch.Tensor:
    """The neural network model is trained based on the configuration parameters,
    evaluated by the loss function, and incrementally adjusted by the optimizer.

    Input:
        `model` (`torch.nn.Module`) : The neural network model to be trained.
        `generator` (`generators.Generator`) : The iterable function that returns
            representations of polymer solution specific viscosity data.
        `optimizer` (`torch.optim.Optimizer`) : Incrementally adjusts the model.
        `loss_fn` (`torch.nn.Module`) : Determines the errors between the true values of
            the features and those predicted by the model.
        `config` (`configuration.NNConfig`) : Defines the attributes of the training
            iterations.

    Returns:
        `avg_loss` (`torch.Tensor`) : The average loss of each feature.
    """

    logger = logging.getLogger("__main__")

    model.train()
    for surfaces, features in generator(num_batches):
        optimizer.zero_grad()

        predicted_features = model(surfaces)
        loss = loss_fn(predicted_features, features)

        loss.mean().backward()
        optimizer.step()

        avg_loss += loss.mean(dim=0)  # Average over the batches
        logger.debug(loss.mean(dim=0))

    avg_loss /= num_batches

    return avg_loss.detach().cpu()


def test(
    model: torch.nn.Module,
    generator: generators.Generator,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_batches: int,
    avg_loss: torch.Tensor,
) -> torch.Tensor:
    """Tests the neural network model based on the configuration parameters using
    the loss function.

    Input:
        `model` (`torch.nn.Module`) : The neural network model to be tested.
        `generator` (`generators.Generator`) : The iterable function that returns
            representations of polymer solution specific viscosity data.
        `loss_fn` (`torch.nn.Module`) : Determines the errors between the true values of
            the features and those predicted by the model.
        `config` (`configuration.NNConfig`) : Defines the attributes of the training
            iterations.

    Returns:
        `avg_loss` (`torch.Tensor`) : The average loss of each feature.
    """

    logger = logging.getLogger("__main__")

    model.eval()
    with torch.no_grad():
        for surfaces, features in generator(num_batches):
            predicted_features = model(surfaces)
            loss = loss_fn(predicted_features, features)

            avg_loss += loss.mean(dim=0)
            logger.debug(loss.mean(dim=0))

    avg_loss /= num_batches

    return avg_loss.cpu()
