from typing import Tuple

import torch

import theoretical_nn_training.data_processing as data
import theoretical_nn_training.generators as gen


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    config: data.Config,
) -> None:

    if config.resolution.eta_sp:
        generator = gen.voxel_image_generator
    else:
        generator = gen.surface_generator

    model.train()

    num_batches = config.train_size // config.batch_size

    for X, y in generator(num_batches, config):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(
    model: torch.nn.Module, loss_fn: torch.nn.Module, config: data.Config
) -> Tuple[torch.Tensor, float]:

    if config.resolution.eta_sp:
        generator = gen.voxel_image_generator
    else:
        generator = gen.surface_generator

    model.eval()

    num_batches = config.test_size // config.batch_size
    avg_loss, avg_error = 0, torch.zeros(3, device=config.device)
    with torch.no_grad():
        for X, y in generator(num_batches, config):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += float(loss.item())
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)

    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_error, avg_loss
