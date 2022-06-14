import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import theoretical_nn_training.generators as generators
import theoretical_nn_training.loss_funcs as loss_funcs
import theoretical_nn_training.models as models
import theoretical_nn_training.training as training
from theoretical_nn_training.configuration import NNConfig


def run(
    config: NNConfig,
    device: torch.device,
    model: torch.nn.Module,
    generator: generators.Generator,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    output_directory: Path,
) -> None:
    """Run a single configuration of settings for the neural network.

    Input:
        `config` (`data.NNConfig`): The configuration used throughout the codebase.
            `NNConfig` is a dataclass (see
            https://docs.python.org/3/library/dataclasses.html), which makes processing
            and reading configurations easier than dictionaries.
        `device` (`torch.device`) : The device on which all operations take place.
        `model` (`torch.nn.Module`) : The neural network model to be trained.
        `generator` (`generators.Generator`) : The iterable function that returns
            representations of polymer solution specific viscosity data.
        `loss_fn` (`torch.nn.Module`) : Determines the errors between the true values of
            the features and those predicted by the model.
        `optimizer` (`torch.optim.Optimizer`) : Incrementally adjusts the model.
        `output_directory` (`pathlib.Path`) : The directory to which the
            results will be saved. Specifically, this includes the loss values
            (`loss_values.csv`) and the model and optimizer state as a binary file
            (`model_and_optimizer`).
    """

    train_errors, test_errors = np.zeros((2, config.epochs, config.layer_sizes[-1]))

    logger = logging.getLogger("__main__")
    logger.info("Running...")

    table_header = " epoch"
    for i in range(config.layer_sizes[-1]):
        table_header += f"  loss_{i}"
    logger.info(table_header)

    for epoch in range(config.epochs):

        # Training
        losses = training.train(
            device=device,
            model=model,
            generator=generator,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
        )
        losses = losses.detach().cpu()
        train_errors[epoch] = losses

        table_entry = f"{epoch:6d}"
        for loss in losses:
            table_entry += f"{loss:8.4f}"
        logger.info(table_entry)

        # Testing
        losses = training.test(
            device=device,
            model=model,
            generator=generator,
            loss_fn=loss_fn,
            config=config,
        )
        losses = losses.cpu()
        test_errors[epoch] = losses

        table_entry = f"{epoch:6d}"
        for loss in losses:
            table_entry += f"{loss:8.4f}"
        logger.info(table_entry)

    # Save loss values
    np.savetxt(
        output_directory / "loss_values.csv",
        np.concatenate((train_errors, test_errors), axis=1),
        fmt="%.6f",
        delimiter=",",
        comments="",
    )

    # Save model and optimizer for testing and further training
    # (test manually only if good results)
    torch.save(
        (model.state_dict(), optimizer.state_dict()),
        output_directory / "model_and_optimizer",
    )


def main() -> None:

    # TODO: add argparse functionality, with -v option for verbose output, -l for
    # logging options, etc.

    # Get configuration file from command line
    if len(sys.argv) != 3:
        raise SyntaxError("Usage: `main.py <config_filename> <output_filename>`")
    config_filename = Path(sys.argv[1])
    output_directory = Path(sys.argv[2])

    if not config_filename.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: `{config_filename.absolute()}`."
        )

    if not output_directory.is_dir():
        print(
            f"Warning: Output directory {output_directory.absolute()} not found."
            " Creating it now."
        )
        output_directory.mkdir(parents=True)

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    current_datetime = datetime.now()
    file_handler = logging.FileHandler(
        output_directory / f"{current_datetime:%Y%m%d-%H%M}.log"
    )
    cli_handler = logging.StreamHandler()

    log_formatter = logging.Formatter(
        fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S"
    )

    file_handler.setFormatter(log_formatter)
    cli_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(cli_handler)

    logger.info("Initializing...")
    config = NNConfig(config_filename)
    logger.debug(f"Read config from {config_filename.absolute()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Set device to {device}")

    logger.debug("Selecting model...")
    if (
        config.channels
        and config.kernel_sizes
        and config.pool_sizes
        and config.resolution.eta_sp
    ):
        model = models.ConvNeuralNet3D(
            resolution=config.resolution,
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            pool_sizes=config.pool_sizes,
            layer_sizes=config.layer_sizes,
        )
        logger.debug("\tInitialized ConvNeuralNet3D")
        generator = generators.VoxelImageGenerator(device=device, config=config)
    elif config.channels and config.kernel_sizes and config.pool_sizes:
        model = models.ConvNeuralNet2D(
            resolution=config.resolution,
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            pool_sizes=config.pool_sizes,
            layer_sizes=config.layer_sizes,
        )
        logger.debug("\tInitialized ConvNeuralNet2D")
        generator = generators.SurfaceGenerator(device=device, config=config)
    else:
        model = models.LinearNeuralNet(
            resolution=config.resolution, layer_sizes=config.layer_sizes
        )
        logger.debug("\tInitialized LinearNeuralNet")
        generator = (
            generators.VoxelImageGenerator(device=device, config=config)
            if config.resolution.eta_sp
            else generators.SurfaceGenerator(device=device, config=config)
        )

    loss_fn = loss_funcs.CustomMSELoss(config.bg_range, config.bth_range)
    logger.debug("Initialized loss functions")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.debug("Initialized optimizer")

    run(
        config=config,
        device=device,
        model=model.to(device),
        generator=generator,
        loss_fn=loss_fn,
        optimizer=optimizer,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()
