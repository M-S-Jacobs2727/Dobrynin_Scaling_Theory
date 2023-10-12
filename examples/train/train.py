"""Run this from the command line as

    python3 train.py <configuration_filename> [-l logfile] [-v] [-h]

where `<configuration_filename>` is a path to a YAML or JSON configuration file (see
examples in the configurations folder). This will create a model, train it on the
generated data, and save the results (model, optimizer, and losses and errors over
training iterations).
"""

import argparse
import logging as log
from typing import Callable, NamedTuple

import numpy as np
import torch

from ...core.configuration import *
from ...core.surface_generator import SurfaceGenerator

from ...models.Inception3 import Inception3
from ...models.Vgg13 import Vgg13


def configure_log(verbosity: int, logfile: str):
    if verbosity == 0:
        level = log.WARNING
    elif verbosity == 1:
        level = log.INFO
    elif verbosity == 2:
        level = log.DEBUG
    else:
        raise ValueError(f"Argument 'verbosity' should be 0, 1, or 2: {verbosity}")

    if logfile:
        log.basicConfig(filename=logfile, format="%(asctime)s - %(levelname)s: %(message)s", level=level)
    else:
        log.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=level)


def parse_args() -> Configuration:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        help="Path to a YAML or JSON configuration file",
    )
    parser.add_argument(
        "-l",
        "--logfile",
        help="Print output to a log file instead of stderr",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print additional information. Can be specified up to twice"
        " (once for basic info, twice for debug info)",
    )

    args = parser.parse_args()

    configure_log(args.verbose, args.logfile)

    config: Configuration = getConfig(args.config_file)
    return config


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
    log.info(f"Saving checkpoint to {filename}")
    log.debug(f"{epoch = }\nmodel: {model.state_dict()}\noptimizer: {optimizer.state_dict()}")
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
    log.info(f"Loading checkpoint from {filename}")
    checkpoint: Checkpoint = torch.load(filename)
    log.debug(f"{checkpoint = }")

    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    
    return checkpoint.epoch


def get_accuracy(train_values: np.ndarray, test_values: np.ndarray) -> float:
    values = np.append(train_values, test_values, axis=0)
    # torch.mean(torch.abs(bth_valid_true-bth_valid_pred)/bth_valid_true)

    bg_acc = np.abs((values[:, 0] - values[:, 3]) / values[:, 0]).mean(axis=0)
    bth_acc = np.abs((values[:, 1] - values[:, 4]) / values[:, 1]).mean(axis=0)
    return (bg_acc + bth_acc) / 2


def train(
    model: torch.nn.Module,
    generator: SurfaceGenerator,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
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

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        old_num_samples = num_samples
        num_samples = num_batches * batch_size
        log.warning(
            f"The number of samples ({old_num_samples})"
            f" is not a multiple of the batch size ({batch_size}),"
            f" and will be reset to {num_samples}."
        )

    idx = int(generator.parameter is ParameterChoice.Bth)
    true_values = np.zeros(num_samples)
    predicted_values = np.zeros_like(true_values)

    model.train()
    for i, (surfaces, *batch_values) in enumerate(generator(num_batches)):
        optimizer.zero_grad()
        pred = model(surfaces)

        loss = loss_fn(pred, batch_values[idx])
        loss.backward()
        optimizer.step()

        true_values[i * batch_size:(i + 1) * batch_size] = batch_values[idx].numpy(force=True)
        predicted_values[i * batch_size:(i + 1) * batch_size] = pred.numpy(force=True)

    return true_values, predicted_values


def validate(
    model: torch.nn.Module,
    generator: SurfaceGenerator,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
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

    batch_size = generator.batch_size
    num_batches = num_samples // batch_size
    if num_batches * batch_size < num_samples:
        old_num_samples = num_samples
        num_samples = num_batches * batch_size
        log.warning(
            f"The number of samples ({old_num_samples})"
            f" is not a multiple of the batch size ({batch_size}),"
            f" and will be reset to {num_samples}."
        )

    true_values = np.zeros((3, num_samples))
    predicted_values = np.zeros(num_samples)

    model.eval()
    with torch.no_grad():
        for i, (surfaces, *batch_values) in enumerate(generator(num_batches)):
            pred = model(surfaces)

            true_values[:, i * batch_size:(i + 1) * batch_size] = batch_values.numpy(force=True)
            predicted_values[i * batch_size:(i + 1) * batch_size] = pred.numpy(force=True)

    return true_values, predicted_values


def run(
    config: Configuration,
    model_Bg: torch.nn.Module,
    model_Bth: torch.nn.Module,
    generator: SurfaceGenerator,
    loss_fn: torch.nn.Module,
    optimizer_Bg: torch.optim.Optimizer,
    optimizer_Bth: torch.optim.Optimizer,
) -> None:
    try:
        device = torch.device(config.device)
        model_Bg.to(device)
        model_Bth.to(device)
    except Exception:
        device = torch.device("cpu")
        model_Bg.to(device)
        model_Bth.to(device)

    best_loss_training = np.zeros((config.train_size, 5))
    best_loss_testing = np.zeros((config.test_size, 5))
    best_acc_training = np.zeros((config.train_size, 5))
    best_acc_testing = np.zeros((config.test_size, 5))

    best_loss = 1e6
    best_acc = 1e6
    best_loss_state = dict()
    best_acc_state = dict()

    for epoch in range(config.epochs):
        train_out: np.ndarray = train(
            model_Bg,
            model_Bth,
            generator,
            optimizer_Bg,
            optimizer_Bth,
            loss_fn,
            config.train_size,
            config.batch_size,
        )

        test_out: np.ndarray = validate(
            model_Bg,
            model_Bth,
            generator,
            loss_fn,
            config.train_size,
            config.batch_size,
        )

        bg_loss = loss_fn(test_out[:, 0], test_out[:, 3])
        bth_loss = loss_fn(test_out[:, 1], test_out[:, 4])
        loss_value = (bg_loss + bth_loss) / 2
        if loss_value < best_loss:
            best_loss = loss_value
            best_loss_state = {
                "epoch": epoch,
                "bg_model_state_dict": model_Bg.state_dict(),
                "bg_optimizer_state_dict": optimizer_Bg.state_dict(),
                "bth_model_state_dict": model_Bth.state_dict(),
                "bth_optimizer_state_dict": optimizer_Bth.state_dict(),
            }
            best_loss_training[: train_out.shape[0]] = train_out
            best_loss_testing[: test_out.shape[0]] = test_out

        acc_value = get_accuracy(train_out, test_out)
        if acc_value < best_acc:
            best_acc = acc_value
            best_acc_state = {
                "epoch": epoch,
                "bg_model_state_dict": model_Bg.state_dict(),
                "bg_optimizer_state_dict": optimizer_Bg.state_dict(),
                "bth_model_state_dict": model_Bth.state_dict(),
                "bth_optimizer_state_dict": optimizer_Bth.state_dict(),
            }
            best_acc_training[: train_out.shape[0]] = train_out
            best_acc_testing[: test_out.shape[0]] = test_out

    end_state = {
        "epoch": epoch,
        "bg_model_state_dict": model_Bg.state_dict(),
        "bg_optimizer_state_dict": optimizer_Bg.state_dict(),
        "bth_model_state_dict": model_Bth.state_dict(),
        "bth_optimizer_state_dict": optimizer_Bth.state_dict(),
    }

    torch.save(best_acc_state, config.output_directory / "best_acc.pt")
    torch.save(best_loss_state, config.output_directory / "best_loss.pt")
    torch.save(end_state, config.output_directory / "end_state.pt")

    np.savez(
        config.output_directory / "best_results.npz",
        best_acc_testing=best_acc_testing,
        best_acc_training=best_acc_training,
        best_loss_testing=best_loss_testing,
        best_loss_training=best_loss_training,
    )


def main():
    config = parse_args()

    generator = SurfaceGenerator(config)
    loss_fn = torch.nn.MSELoss()

    run(config, model_Bg, model_Bth, generator, loss_fn, optimizer_Bg, optimizer_Bth)


if __name__ == "__main__":
    main()
