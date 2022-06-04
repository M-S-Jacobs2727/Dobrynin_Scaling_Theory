from typing import Callable, Tuple

import torch

from theoretical_nn_training.datatypes import Resolution


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    resolution: Resolution,
    generator: Callable,
    optimizer: torch.optim.Optimizer,
) -> None:

    model.train()

    num_batches = num_samples // batch_size

    for X, y in generator(num_batches, batch_size, device, resolution):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    resolution: Resolution,
    generator: Callable,
) -> Tuple[torch.Tensor, float]:

    model.eval()

    num_batches = num_samples // batch_size
    avg_loss, avg_error = 0, torch.zeros(3, device=device)
    with torch.no_grad():
        for X, y in generator(num_batches, batch_size, device, resolution):
            pred = model(X)
            loss = loss_fn(pred, y)

            avg_loss += float(loss.item())
            avg_error += torch.mean(torch.abs(y - pred) / y, 0)

    avg_loss /= num_batches
    avg_error /= num_batches

    return avg_error, avg_loss
