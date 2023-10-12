import logging
from typing import NamedTuple, Callable, Protocol

import numpy as np
import torch

from psst.surface_generator import SurfaceGenerator
from psst.configuration import *


class Model(Protocol):
    def __call__(self, inp: torch.Tensor) -> torch.Tensor:
        ...
    
    def train(self) -> None:
        ...
    
    def eval(self) -> None:
        ...


class Checkpoint(NamedTuple):
    epoch: int
    model_state: dict
    optimizer_state: dict


def save_checkpoint(
        filename: str,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
):
    checkpoint = Checkpoint(
        epoch,
        model.state_dict(),
        optimizer.state_dict(),
    )
    torch.save(checkpoint, filename)


def load_checkpoint(
        filename: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint: Checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    
    return checkpoint.epoch


def train(
    model: Model,
    generator: SurfaceGenerator,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_samples: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """The neural network model is trained based on the configuration parameters,
    evaluated by the loss function, and incrementally adjusted by the optimizer.

    Input:
    - `model` (`torch.nn.Module`) : The neural network model.
    - `generator` (`generators.Generator`) : The iterable function that returns
    representations of polymer solution specific viscosity data.
    - `optimizer` (`torch.optim.Optimizer`) : Incrementally adjusts the model.
    - `loss_fn` (`torch.nn.Module`) : Determines the errors between the true values of
    the features and those predicted by the model.
    - `num_samples` (`int`): The number of sample surfaces to run through training.

    Returns:
    - `true_values` (`torch.Tensor`) : A 1D tensor with `num_samples` elements containing
    the true (generated) values of the parameter to predict.
    - `predicted_values` (`torch.Tensor`) : A 1D tensor with `num_samples` elements containing
    the predicted values from the model.
    """

    log = logging.getLogger("psst.main")

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        log.warn(
            "num_samples (%d) is not evenly divisible by batch_size (%d)."
            "\nResetting num_samples to %d.", num_samples, batch_size, num_batches*batch_size)
        num_samples = num_batches * batch_size

    choice = int(generator.parameter is ParameterChoice.Bth)
    avg_loss: float = 0.0
    true_values = np.zeros(num_samples)
    predicted_values = np.zeros_like(true_values)

    model.train()
    count = 0
    log.info("Starting training run of %d batches", num_batches)
    for surfaces, *batch_values in generator(num_batches):
        optimizer.zero_grad()
        log.debug("Training batch %d", count / batch_size)
        pred = model(surfaces)

        log.debug("Computing loss")
        loss = loss_fn(pred, batch_values[choice])
        loss.backward()
        avg_loss += loss
        optimizer.step()

        log.debug("Storing results")
        true_values[count:count + batch_size] = batch_values[choice].numpy(force=True)
        predicted_values[count:count + batch_size] = pred.numpy(force=True)
        count += batch_size

    return avg_loss / num_batches, true_values, predicted_values


def validate(
    model: Model,
    generator: SurfaceGenerator,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_samples: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Tests the neural network model based on the configuration parameters using
    the loss function.

    Input:
    - `model` (`torch.nn.Module`) : The neural network model.
    - `generator` (`generators.Generator`) : The iterable function that returns
    representations of polymer solution specific viscosity data.
    - `num_samples` (`int`): The number of sample surfaces to run through validation.

    Returns:
    - `true_values` (`torch.Tensor`) : A 1D tensor with `num_samples` elements containing
    the true (generated) values of the parameter to predict.
    - `predicted_values` (`torch.Tensor`) : A 1D tensor with `num_samples` elements containing
    the predicted values from the model.
    """

    log = logging.getLogger("psst.main")

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        log.warn(
            "num_samples (%d) is not evenly divisible by batch_size (%d)."
            "\nResetting num_samples to %d.", num_samples, batch_size, num_batches*batch_size)
        num_samples = num_batches * batch_size

    choice = int(generator.parameter is ParameterChoice.Bth)
    avg_loss: float = 0.0
    true_values = np.zeros((3, num_samples))
    predicted_values = np.zeros(num_samples)

    model.eval()
    count = 0
    log.info("Starting validation run of %d batches", num_batches)
    with torch.no_grad():
        for surfaces, *batch_values in generator(num_batches):
            log.debug("Testing batch %d", count / batch_size)
            pred = model(surfaces)
            
            log.debug("Computing loss")
            loss = loss_fn(pred, batch_values[choice])
            avg_loss += loss

            log.debug("Storing results")
            true_values[0, count:count + batch_size] = batch_values[0].numpy(force=True)
            true_values[1, count:count + batch_size] = batch_values[1].numpy(force=True)
            true_values[2, count:count + batch_size] = batch_values[2].numpy(force=True)
            predicted_values[count:count + batch_size] = pred.numpy(force=True)
            count += batch_size

    return avg_loss / num_batches, true_values, predicted_values
