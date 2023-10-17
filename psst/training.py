import logging
from pathlib import Path
from typing import NamedTuple

import torch

from psst.samplegenerator import SampleGenerator


class Checkpoint(NamedTuple):
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

    :param model: PyTorch ML model to be validated
    :type model: torch.nn.Module
    :param optimizer: PyTorch torch.optim.Optimizer used to train the model
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: PyTorch loss function to evaluate model accuracy
    :type loss_fn: torch.nn.Module
    :param generator: Procedurally generates data for model training
    :type generator: psst.samplegenerator.SampleGenerator
    :param num_samples: Number of samples to generate and train
    :type num_samples: int
    :return: Average loss over training cycle
    :rtype: float
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

    :param model: PyTorch ML model to be validated
    :type model: torch.nn.Module
    :param loss_fn: PyTorch loss function to evaluate model accuracy
    :type loss_fn: torch.nn.Module
    :param generator: Procedurally generates data for model evaluation
    :type generator: psst.samplegenerator.SampleGenerator
    :param num_samples: Number of samples to generate and validate
    :type num_samples: int
    :return: Average loss over validation cycle
    :rtype: float
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
    num_epochs: int,
    num_samples_train: int,
    num_samples_test: int,
    checkpoint_filename: str | Path,
    checkpoint_frequency: int,
):
    """Run the model through `num_epochs` train/test cycles.

    :param model: PyTorch ML model to be validated
    :type model: torch.nn.Module
    :param optimizer: PyTorch torch.optim.Optimizer used to train the model
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: PyTorch loss function to evaluate model accuracy
    :type loss_fn: torch.nn.Module
    :param generator: Procedurally generates data for model training
    :type generator: psst.samplegenerator.SampleGenerator
    :param num_epochs: Number of train/test cycles to run
    :type num_epochs: int
    :param num_samples_train: Number of samples to generate during a single training
        session
    :type num_samples_train: int
    :param num_samples_test: Number of samples to generate during a single validation
        session
    :type num_samples_test: int
    :param checkpoint_filename: Filename to save the model and optimizer to at a given
        checkpoint
    :type checkpoint_filename: str | Path
    :param checkpoint_frequency: If positive, save the model and optimizer every this
        many epochs. If negative, save the model and optimizer when the loss reaches
        a new minimum. If zero, do not checkpoint.
    :type checkpoint_frequency: int
    """
    chkpt = Checkpoint(0, model.state_dict(), optimizer.state_dict())
    min_loss = 1e6

    for epoch in range(num_epochs):
        train(model, optimizer, loss_fn, generator, num_samples_train)
        loss = validate(model, loss_fn, generator, num_samples_test)
        if (checkpoint_frequency < 0 and loss < min_loss) or (
            checkpoint_frequency > 0 and epoch % checkpoint_frequency == 0
        ):
            min_loss = loss
            chkpt.epoch = epoch
            chkpt.model_state = model.state_dict()
            chkpt.optimizer_state = optimizer.state_dict()

    torch.save(chkpt, checkpoint_filename)
