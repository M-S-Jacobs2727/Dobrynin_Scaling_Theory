from __future__ import annotations
import logging
from pathlib import Path
from typing import NamedTuple

import torch

from psst.samplegenerator import SampleGenerator


class Checkpoint(NamedTuple):
    """Represents a state during training. Can be easily saved to file `filepath` with

    >>> chkpt = psst.Checkpoint(epoch, model.state_dict(), optimizer.state_dict())
    >>> torch.save(chkpt, filepath)

    Attributes:
        epoch (int): How many cycles of training have been completed.
        model_state (dict): The state of the neural network model as given by
          torch.nn.Module.state_dict()
        optimizer_state (dict): The state of the training optimizer as given by
          torch.optim.Optimizer.state_dict()
    """
    epoch: int
    model_state: dict
    optimizer_state: dict


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    generator: SampleGenerator,
    num_samples: int,
) -> float:
    """The neural network model is trained based on the configuration parameters,
    evaluated by the loss function, and incrementally adjusted by the optimizer.

    Args:
        model (torch.nn.Module): Machine learning model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to train the model.
        loss_fn (torch.nn.Module): Loss function to evaluate model accuracy
        generator (SampleGenerator): Procedurally generates data for model training.
        num_samples (int): Number of samples to generate and train.

    Returns:
        float: Average loss over training cycle.
    """

    log = logging.getLogger("psst.main")

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        log.warn(
            "num_samples (%d) is not evenly divisible by batch_size (%d)."
            "\nResetting num_samples to %d.",
            num_samples,
            batch_size,
            num_batches * batch_size,
        )
        num_samples = num_batches * batch_size

    choice = 1 if generator.parameter == "Bth" else 0
    avg_loss: float = 0.0

    model.train()
    count = 0
    log.info("Starting training run of %d batches", num_batches)
    for samples, *batch_values in generator(num_batches):
        optimizer.zero_grad()
        log.debug("Training batch %d", count / batch_size)
        pred: torch.Tensor = model(samples)

        log.debug("Computing loss")
        loss: torch.Tensor = loss_fn(pred, batch_values[choice])
        loss.backward()
        avg_loss += loss
        optimizer.step()

    return avg_loss / num_batches


def validate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    generator: SampleGenerator,
    num_samples: int,
) -> float:
    """Tests the neural network model based on the configuration parameters using
    the loss function.

    Args:
        model (torch.nn.Module): Machine learning model to be validated.
        loss_fn (torch.nn.Module): Loss function to evaluate model accuracy
        generator (SampleGenerator): Procedurally generates data for model validation.
        num_samples (int): Number of samples to generate and validate.

    Returns:
        float: Average loss over validation cycle.
    """

    log = logging.getLogger("psst.main")

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        log.warn(
            "num_samples (%d) is not evenly divisible by batch_size (%d)."
            "\nResetting num_samples to %d.",
            num_samples,
            batch_size,
            num_batches * batch_size,
        )
        num_samples = num_batches * batch_size

    choice = 0 if generator.parameter == "Bg" else 1
    avg_loss: float = 0.0

    model.eval()
    log.info("Starting validation run of %d batches", num_batches)
    with torch.no_grad():
        for i, (samples, *batch_values) in enumerate(generator(num_batches)):
            log.debug("Testing batch %d", i)
            pred = model(samples)

            log.debug("Computing loss")
            loss = loss_fn(pred, batch_values[choice])
            avg_loss += loss

    return avg_loss / num_batches


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    generator: SampleGenerator,
    *,
    start_epoch: int = 0,
    total_num_epochs: int,
    num_samples_train: int,
    num_samples_test: int,
    checkpoint_filename: str | Path,
    checkpoint_frequency: int,
):
    """Run the model through `num_epochs` train/test cycles.

    Args:
        model (torch.nn.Module): Machine learning model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer used to train the model.
        loss_fn (torch.nn.Module): Loss function to evaluate model accuracy
        generator (SampleGenerator): Procedurally generates data for model training.
    
    Keyword Arguments:
        start_epoch (int): The index of the first epoch to run (useful for continuing
          training from a checkpoint).
        total_num_epochs (int): Total number of training/testing cycles to run.
        num_samples_train (int): Number of samples to generate during a single
          training session.
        num_samples_test (int): Number of samples to generate during a single
          validation session.
        checkpoint_filename (str | Path): Filename to save the model and optimizer to
          at a given checkpoint. Note: The convention is to save PyTorch models with
          the '.pt' extension.
        checkpoint_frequency (int): If positive, save the model and optimizer every
          this many epochs. If negative, save the model and optimizer when the loss
          reaches a new minimum. If zero, do not checkpoint.
    """
    chkpt = Checkpoint(start_epoch, model.state_dict(), optimizer.state_dict())
    min_loss = 1e6

    for epoch in range(start_epoch, total_num_epochs):
        train(model, optimizer, loss_fn, generator, num_samples_train)
        loss = validate(model, loss_fn, generator, num_samples_test)
        if (checkpoint_frequency < 0 and loss < min_loss) or (
            checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0
        ):
            min_loss = loss
            chkpt.epoch = epoch
            chkpt.model_state = model.state_dict()
            chkpt.optimizer_state = optimizer.state_dict()

    if checkpoint_frequency != 0:
        torch.save(chkpt, checkpoint_filename)
