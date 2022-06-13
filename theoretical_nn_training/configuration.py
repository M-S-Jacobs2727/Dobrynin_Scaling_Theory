import json
import logging
from pathlib import Path

import yaml

from theoretical_nn_training.data_processing import Range, Resolution


class NNConfig:
    def __init__(self, config_filename: Path):
        logger = logging.getLogger(__name__)
        # Load configuration file and assign to config_dictionary
        extension = config_filename.suffix
        with open(config_filename, "r") as f:
            if extension == ".yaml":
                config_dict = dict(yaml.safe_load(f))
            elif extension == ".json":
                config_dict = dict(json.load(f))
            else:
                raise SyntaxError(f"Invalid file extension: {extension}")

        # Optimizer learning rate
        self.learning_rate: float = config_dict["learning_rate"]
        logger.debug(f"Loaded {self.learning_rate = :.5f}.")

        # Min and max values of concentration, Nw, and viscosity
        self.phi_range = Range(
            config_dict["phi_range"]["min"], config_dict["phi_range"]["max"]
        )
        self.nw_range = Range(
            config_dict["nw_range"]["min"], config_dict["nw_range"]["max"]
        )
        self.eta_sp_range = Range(
            config_dict["eta_sp_range"]["min"],
            config_dict["eta_sp_range"]["max"],
        )
        logger.debug(
            f"Loaded {self.phi_range = }, {self.nw_range = }, {self.eta_sp_range = }."
        )

        # Min, max, and distribution definitions for Bg, Bth, and Pe
        self.bg_range = Range(
            config_dict["bg_range"]["min"], config_dict["bg_range"]["max"]
        )
        self.bth_range = Range(
            config_dict["bth_range"]["min"], config_dict["bth_range"]["max"]
        )
        self.pe_range = Range(
            config_dict["pe_range"]["min"], config_dict["pe_range"]["max"]
        )
        logger.debug(
            f"Loaded {self.bg_range = }, {self.bth_range = }, {self.pe_range = }."
        )

        # Check for invalid combinations of alpha, beta, mu, and sigma
        for param_range, param_dictionary in zip(
            [self.bg_range, self.bth_range, self.pe_range],
            [
                config_dict["bg_range"],
                config_dict["bth_range"],
                config_dict["pe_range"],
            ],
        ):
            keys = param_dictionary.keys()
            if "alpha" in keys and "beta" in keys and "mu" in keys and "sigma" in keys:
                raise SyntaxError(
                    "Only one pair of (alpha, beta) or (mu, sigma) may be specified."
                    f" Instead, both are specified in {param_dictionary}"
                )
            if ("alpha" in keys) ^ ("beta" in keys):
                logger.warn("Only one of alpha/beta is specified. Ignoring.")
            if ("mu" in keys) ^ ("sigma" in keys):
                logger.warn("Only one of mu/sigma is specified. Ignoring.")
            if "alpha" in keys and "beta" in keys:
                param_range.alpha = param_dictionary["alpha"]
                param_range.beta = param_dictionary["beta"]
                logger.debug(f"Loaded {param_range.alpha = }, {param_range.beta = }.")
            elif "mu" in keys and "sigma" in keys:
                param_range.mu = param_dictionary["mu"]
                param_range.sigma = param_dictionary["sigma"]
                logger.debug(f"Loaded {param_range.mu = }, {param_range.sigma = }.")
            else:
                param_range.alpha = None
                param_range.beta = None
                param_range.mu = None
                param_range.sigma = None
                logger.debug("Only min and max loaded.")

        # Model training parameters
        self.batch_size: int = config_dict["batch_size"]
        self.train_size: int = config_dict["train_size"]
        self.test_size: int = config_dict["test_size"]
        self.epochs: int = config_dict["epochs"]
        logger.debug(
            f"Loaded {self.batch_size = }, {self.train_size = }, {self.test_size = },"
            f" and {self.epochs = }."
        )

        # Surface resolution
        self.resolution = Resolution(*config_dict["resolution"])
        logger.debug(f"Loaded {self.resolution = }.")

        # Model
        model_dict = dict(config_dict["model"])

        # Number of nodes in each linear NN layer
        self.layer_sizes = tuple(model_dict["layer_sizes"])
        logger.debug(f"Loaded {self.layer_sizes = }.")

        # Define these three hyperparameters in the configuration file with equal
        # lengths to specify a convolutional neural network.
        if (
            "channels" in model_dict.keys()
            and "kernel_sizes" in model_dict.keys()
            and "pool_sizes" in model_dict.keys()
        ):
            self.channels = tuple(model_dict["channels"])
            self.kernel_sizes = tuple(model_dict["kernel_sizes"])
            self.pool_sizes = tuple(model_dict["pool_sizes"])
            if not (
                len(self.channels) == len(self.kernel_sizes) == len(self.pool_sizes)
            ):
                raise ValueError(
                    "Expected configuration parameters `channels`, `kernel_sizes`, and"
                    "`pool_sizes` to have equal length. Instead, the lengths are"
                    f" {len(self.channels)}, {len(self.kernel_sizes)}, and"
                    f" {len(self.pool_sizes)}, respectively."
                )
            logger.debug(
                f"Loaded {self.channels = }, {self.kernel_sizes = }, and"
                f" {self.pool_sizes = }."
            )
        elif (
            "channels" in model_dict.keys()
            or "kernel_sizes" in model_dict.keys()
            or "pool_sizes" in model_dict.keys()
        ):
            logger.warn(
                "Not all of channels/kernel_sizes/pool_sizes are specified."
                " Ignoring."
            )
            self.channels = None
            self.kernel_sizes = None
            self.pool_sizes = None
        else:
            self.channels = None
            self.kernel_sizes = None
            self.pool_sizes = None
            logger.debug("Channels, kernel sizes, and pool sizes are not specified.")
