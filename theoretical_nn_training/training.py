import logging
from typing import Callable, Tuple

import torch

import theoretical_nn_training.generators as generators
import theoretical_nn_training.loss_funcs as loss_funcs
from theoretical_nn_training.configuration import NNConfig


def train(
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: NNConfig,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, float]:
    """Trains the neural network model based on the configuration parameters, evaluated
    by the loss function, and incrementally adjusted by the optimizer. All operations
    take place on the device (e.g., CUDA-enabled GPU).
    """

    if config.resolution.eta_sp:
        generator = generators.voxel_image_generator
        logger.debug(
            "Training on 3D visualization of surfaces with"
            " `generators.voxel_image_generator`."
        )
    else:
        generator = generators.surface_generator
        logger.debug(
            "Training on 2D visualization of surfaces with"
            " `generators.surface_generator`."
        )

    model.train()
    loss2 = loss_funcs.CustomMSELoss(config.bg_range, config.bth_range, mode="none")

    num_batches = config.train_size // config.batch_size
    avg_loss, avg_error = 0.0, torch.zeros(config.layer_sizes[-1], device=device)
    logger.debug(
        f"Beginning training on {config.train_size} samples with batch size"
        f" {config.batch_size}."
    )
    for X, y in generator(num_batches, device, config):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()
        avg_error += loss2(pred, y)

    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_error, avg_loss


def test(
    device: torch.device,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: NNConfig,
    logger: logging.Logger,
) -> Tuple[torch.Tensor, float]:
    """Tests the neural network model based on the configuration parameters using
    the loss function. All operations take place on the device (e.g., CUDA-enabled GPU).
    """

    if config.resolution.eta_sp:
        generator = generators.voxel_image_generator
        logger.debug(
            "Testing on 3D visualization of surfaces with"
            " `generators.voxel_image_generator`."
        )
    else:
        generator = generators.surface_generator
        logger.debug(
            "Testing on 2D visualization of surfaces with"
            " `generators.surface_generator`."
        )

    model.eval()
    loss2 = loss_funcs.CustomMSELoss(config.bg_range, config.bth_range, mode="none")

    num_batches = config.test_size // config.batch_size
    avg_loss, avg_error = 0.0, torch.zeros(config.layer_sizes[-1], device=device)
    logger.debug(
        f"Beginning testing on {config.test_size} samples with batch size"
        f" {config.batch_size}."
    )
    with torch.no_grad():
        for X, y in generator(num_batches, device, config):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += loss.item()
            avg_error += loss2(pred, y)

    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_error, avg_loss
