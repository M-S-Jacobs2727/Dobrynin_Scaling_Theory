import argparse
import logging
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
    model: torch.nn.Module,
    generator: generators.Generator,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> None:
    """Run a single configuration of settings for the neural network.

    Input:
        `config` (`data.NNConfig`): The configuration used throughout the codebase.
            `NNConfig` is a dataclass (see
            https://docs.python.org/3/library/dataclasses.html), which makes processing
            and reading configurations easier than dictionaries.
        `model` (`torch.nn.Module`) : The neural network model to be trained.
        `generator` (`generators.Generator`) : The iterable function that returns
            representations of polymer solution specific viscosity data.
        `loss_fn` (`torch.nn.Module`) : Determines the errors between the true values of
            the features and those predicted by the model.
        `optimizer` (`torch.optim.Optimizer`) : Incrementally adjusts the model.
    """

    train_errors, test_errors = np.zeros((2, config.epochs, config.layer_sizes[-1]))

    logger = logging.getLogger("__main__")
    logger.info("Running...")

    table_header = " epoch"
    for i in range(config.layer_sizes[-1]):
        table_header += f"    loss_{i}"
    logger.info(table_header)

    for epoch in range(config.epochs):

        # Training
        # losses will be the individual loss of each feature, as a 1D tensor with
        # length config.layer_sizes[-1]
        losses = training.train(
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
            table_entry += f"  {loss:8.4f}"
        logger.info(table_entry)

        # Testing
        losses = training.test(
            model=model,
            generator=generator,
            loss_fn=loss_fn,
            config=config,
        )
        losses = losses.cpu()
        test_errors[epoch] = losses

        table_entry = f"{epoch:6d}"
        for loss in losses:
            table_entry += f"  {loss:8.4f}"
        logger.info(table_entry)

    # Save loss values
    np.savetxt(
        config.output_directory / "loss_values.csv",
        np.concatenate((train_errors, test_errors), axis=1),
        fmt="%.6f",
        delimiter=",",
        comments="",
    )

    # Save model and optimizer for testing and further training
    # (test manually only if good results)
    torch.save(
        (model.state_dict(), optimizer.state_dict()),
        config.output_directory / "model_and_optimizer",
    )


def main() -> None:

    # Parse arguments for logfile and verbosity (logging.debug vs. logging.info)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", type=str)
    parser.add_argument(
        "-l", "--logfile", type=str, help="If unspecified, will log to console"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Get configuration file from command line
    config_filename = Path(args.config_filename)

    if not config_filename.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: `{config_filename.absolute()}`."
        )

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.logfile:
        handler = logging.FileHandler(args.logfile)
    else:
        handler = logging.StreamHandler()

    log_formatter = logging.Formatter(
        fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S"
    )

    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    logger.info("Initializing...")
    config = NNConfig(config_filename)
    logger.debug(f"Read config from {config_filename.absolute()}")

    # Model and generator
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
        generator = generators.VoxelImageGenerator(config=config)
        logger.debug("\tInitialized ConvNeuralNet3D with VoxelImageGenerator")
    elif config.channels and config.kernel_sizes and config.pool_sizes:
        model = models.ConvNeuralNet2D(
            resolution=config.resolution,
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            pool_sizes=config.pool_sizes,
            layer_sizes=config.layer_sizes,
        )
        generator = generators.SurfaceGenerator(config=config)
        logger.debug("\tInitialized ConvNeuralNet2D with SurfaceGenerator")
    else:
        model = models.LinearNeuralNet(
            resolution=config.resolution, layer_sizes=config.layer_sizes
        )
        if config.resolution.eta_sp:
            generator = generators.VoxelImageGenerator(config=config)
            logger.debug("\tInitialized LinearNeuralNet with VoxelImageGenerator.")
        else:
            generator = generators.SurfaceGenerator(config=config)
            logger.debug("\tInitialized LinearNeuralNet with SurfaceGenerator.")

    # Loss function
    if config.layer_sizes[-1] == 3:
        loss_fn = loss_funcs.CustomMSELoss(
            config.bg_range, config.bth_range, config.pe_range, mode="none"
        )
    elif config.layer_sizes[-1] == 2:
        loss_fn = loss_funcs.CustomMSELoss(
            config.bg_range, config.bth_range, mode="none"
        )
    else:
        raise RuntimeError(
            "Configuration parameter layer_sizes must end in either 2 or 3, not"
            f" {config.layer_sizes[-1]}."
        )
    logger.debug(f"Initialized loss function for {config.layer_sizes[-1]} features.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.debug(f"Initialized optimizer with learning rate {config.learning_rate}.")

    # Run
    run(
        config=config,
        model=model.to(config.device),
        generator=generator,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
