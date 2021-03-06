"""Run this from the command line as

    python3 main.py <configuration_filename> [-l logfile] [-v] [-h]

where `<configuration_filename>` is a path to a YAML or JSON configuration file (see
examples in the configurations folder). This will create a model, train it on the
generated data, and save the results (model, optimizer, and losses and errors over
training iterations).
"""

import argparse
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import theoretical_nn_training.configuration as configuration
import theoretical_nn_training.generators as generators
import theoretical_nn_training.models as models
import theoretical_nn_training.training as training
from theoretical_nn_training.data_processing import Mode


def run(
    config: configuration.NNConfig,
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

    num_features = 3 if config.mode is Mode.MIXED else 2
    train_losses, test_losses = np.zeros((2, config.epochs, num_features))

    logger = logging.getLogger("__main__")
    logger.info("Running...")

    table_header = " epoch"
    if config.mode is not Mode.THETA:
        table_header += "   Bg_loss"
    if config.mode is not Mode.GOOD:
        table_header += "  Bth_loss"
    table_header += "   Pe_loss"
    logger.info(table_header)

    for epoch in range(config.epochs):

        # Training
        # losses will be the individual loss of each feature, as a 1D tensor with
        # length config.layer_sizes[-1]
        train_losses[epoch] = training.train(
            model=model,
            generator=generator,
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_batches=config.train_size // config.batch_size,
            avg_loss=torch.zeros(num_features, device=config.device),
        ).numpy()

        table_entry = f"{epoch+1:6d}"
        for loss in train_losses[epoch]:
            table_entry += f"  {loss:8.4f}"
        logger.info(table_entry)

        # Testing
        test_losses[epoch] = training.test(
            model=model,
            generator=generator,
            loss_fn=loss_fn,
            num_batches=config.train_size // config.batch_size,
            avg_loss=torch.zeros(num_features, device=config.device),
        ).numpy()

        table_entry = f"{epoch+1:6d}"
        for loss in test_losses[epoch]:
            table_entry += f"  {loss:8.4f}"
        logger.info(table_entry)

    # Save loss values
    loss_file = config.output_directory / "loss_values.csv"
    np.savetxt(
        loss_file,
        np.concatenate((train_losses, test_losses), axis=1),
        fmt="%.6f",
        delimiter=",",
        comments="",
    )
    logger.info(f"Saved loss progress in {loss_file}")

    # Save model and optimizer for testing and further training
    # (test manually only if good results)
    model_file = config.output_directory / "model_and_optimizer"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        model_file,
    )
    logger.info(f"Saved model and optimizer in {model_file}")
    logger.info("Done.\n")


def main() -> None:

    # Parse arguments for logfile and verbosity (logging.debug vs. logging.info)
    parser = argparse.ArgumentParser()
    parser.add_argument("config_filename", type=str)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Log all debug messages."
    )
    parser.add_argument(
        "-m",
        "--modelfile",
        type=str,
        help="The location of a file containing the states of the model and optimizer"
        " as a dictionary. To be read by `torch.load`.",
    )
    args = parser.parse_args()

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    handler = logging.StreamHandler()

    log_formatter = logging.Formatter(
        fmt="%(asctime)s: %(message)s", datefmt="%H:%M:%S"
    )

    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    logger.info("Initializing...")

    # Get configuration file from command line
    config_filename = Path(args.config_filename)
    config = configuration.read_config_from_file(config_filename)
    logger.debug(f"Read config from {config_filename.absolute()}")

    # Model
    logger.debug("Selecting model...")
    if (
        config.channels
        and config.kernel_sizes
        and config.pool_sizes
        and config.resolution.eta_sp
    ):
        model = models.ConvNeuralNet3D(
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            pool_sizes=config.pool_sizes,
            layer_sizes=config.layer_sizes,
        )
        logger.debug("\tInitialized ConvNeuralNet3D")
    elif config.channels and config.kernel_sizes and config.pool_sizes:
        model = models.ConvNeuralNet2D(
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            pool_sizes=config.pool_sizes,
            layer_sizes=config.layer_sizes,
        )
        logger.debug("\tInitialized ConvNeuralNet2D")
    else:
        model = models.LinearNeuralNet(layer_sizes=config.layer_sizes)
        logger.debug("\tInitialized LinearNeuralNet.")

    model = model.to(config.device)

    # Generator
    if config.resolution.eta_sp:
        generator = generators.VoxelImageGenerator(config=config)
        logger.debug("Initialized VoxelImageGenerator.")
    else:
        generator = generators.SurfaceGenerator(config=config)
        logger.debug("Initialized SurfaceGenerator.")

    # Loss function
    loss_fn = torch.nn.MSELoss(reduction="none")
    logger.debug("Initialized loss function.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.debug(f"Initialized optimizer with learning rate {config.learning_rate}.")

    # If the -m argument is passed with a valid filename, the model and optimizer states
    # will be loaded from that file. If the filename is invalid, an error will be
    # raised.
    if args.modelfile:
        if not Path(args.modelfile).is_file():
            raise FileNotFoundError(
                logger.exception(f"File {args.modelfile} not found.")
            )
        checkpoint = torch.load(args.modelfile)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Loaded model and optimizer from checkpoint successfully.")

    # Run
    run(
        config=config,
        model=model,
        generator=generator,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )


if __name__ == "__main__":
    main()
