import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import theoretical_nn_training.loss_funcs as loss_funcs
import theoretical_nn_training.models as models
import theoretical_nn_training.training as training
from theoretical_nn_training.configuration import NNConfig


def run(
    config: NNConfig,
    device: torch.device,
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    logger: logging.Logger,
    output_directory: Path,
) -> None:
    """Run a single configuration of settings for the neural network.

    Input:
        `config` (`data.NNConfig`): The configuration used throughout the codebase.
            `NNConfig` is a dataclass (see
            https://docs.python.org/3/library/dataclasses.html), which makes processing
            and reading configurations easier than dictionaries.
    """
    # TODO add additional input comments in docs
    logger.info("Running...")

    logger.info(" epoch  loss    bg_err  bth_err ")
    train_errors, test_errors = np.zeros((2, config.epochs, 1 + config.layer_sizes[-1]))
    for epoch in range(config.epochs):

        error_ratios, loss = training.train(
            device=device,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            logger=logger,
        )
        bg_error, bth_error = error_ratios.cpu()[:2]
        logger.info(f"{epoch:6d}{loss:8.4f}{bg_error:8.4f}{bth_error:8.4f}")
        train_errors[epoch] = [loss, bg_error.detach(), bth_error.detach()]

        error_ratios, loss = training.test(
            device=device, model=model, loss_fn=loss_fn, config=config, logger=logger
        )
        bg_error, bth_error = error_ratios.cpu()[:2]
        logger.info(f"{epoch:6d}{loss:8.4f}{bg_error:8.4f}{bth_error:8.4f}")
        test_errors[epoch] = [loss, bg_error, bth_error]

        # The error ratios are defined in training.py as
        # abs(expected - predicted)/expected for each of bg_param, bth_param, and
        # pe_param, where expected is the true value of the parameter and predicted is
        # the value returned by the NN model.

    # Save loss and error values
    np.savetxt(
        output_directory / "loss_values.csv",
        np.stack((train_errors, test_errors), axis=0),
        fmt="%.6f",
        delimiter=",",
        header="train_loss,train_bg_error,train_bth_error,"
        "test_loss,test_bg_error,test_bth_error",
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
    logger = logging.getLogger("main")
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
    logger.debug(f"Read config from {config_filename.absolute()}:")
    logger.debug(f"{config}")

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
    elif config.channels and config.kernel_sizes and config.pool_sizes:
        model = models.ConvNeuralNet2D(
            resolution=config.resolution,
            channels=config.channels,
            kernel_sizes=config.kernel_sizes,
            pool_sizes=config.pool_sizes,
            layer_sizes=config.layer_sizes,
        )
        logger.debug("\tInitialized ConvNeuralNet2D")
    else:
        model = models.LinearNeuralNet(
            resolution=config.resolution, layer_sizes=config.layer_sizes
        )
        logger.debug("\tInitialized LinearNeuralNet")

    loss_fn = loss_funcs.CustomMSELoss(config.bg_range, config.bth_range)
    logger.debug("Initialized loss function")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logger.debug("Initialized optimizer")

    run(config, device, model.to(device), loss_fn, optimizer, logger, output_directory)


if __name__ == "__main__":
    main()
