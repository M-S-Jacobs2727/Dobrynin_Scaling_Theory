import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml

from theoretical_nn_training.data_processing import Range, Resolution


@dataclass
class NNConfig:
    learning_rate: float
    phi_range: Range
    nw_range: Range
    eta_sp_range: Range
    bg_range: Range
    bth_range: Range
    pe_range: Range
    batch_size: int
    train_size: int
    test_size: int
    epochs: int
    layer_sizes: Tuple[int, ...]
    channels: Optional[Tuple[int, ...]] = None
    kernel_sizes: Optional[Tuple[int, ...]] = None
    pool_sizes: Optional[Tuple[int, ...]] = None

    def __init__(self, config_filename: Union[Path, str]):
        """Configuration dataclass. Reads a YAML or JSON configuration file and sets
        corresponding attributes for the returned object. The attributes listed below
        are read from the configuration file with the same hierarchical structure.

        Attributes:

        `learning_rate` (`float`) : The learning rate of the PyTorch optimizer Adam.

        `xxx_range` (`data_processing.Range`) : These objects define the minimum
            (`min`), maximum (`max`), and distribution (`alpha` and `beta` for the
            Beta distribution, `mu` and `sigma` for the LogNormal distribution) of
            their respective parameters.
        `phi_range` : The desired range of concentrations.
        `nw_range` : The desired range of weight-average degrees of polymerization.
        `eta_sp_range` : The desired range of specific viscosity.
        `bg_range` : The desired range and distribution settings for the parameter
            $B_g$.
        `bth_range` : The desired range and distribution settings for the parameter
            $B_{th}$.
        `Pe_range` : The desired range and distribution settings for the packing
            number $P_e$.

        `batch_size` (`int`) : The number of samples given to the model per batch.
        `train_size` (`int`) : The number of samples given to the model over one
            training iteration.
        `test_size` (`int`) : The number of samples given to the model over one
            testing iteration.
        `epochs` (`int`) : The number of training/testing iterations.

        `model` contains the details of the desired neural network architecture.
        If the model is a convolutional neural network, then each of `channels`,
        `kernel_sizes`, and `pool_sizes` must be specified and have equal lengths.
        The architecture of the convolutional layers is a sequence of (convolution,
        ReLU activation layer, max-pool) sequences, the result of which is fed into
        a linear neural network.
            `layer_sizes` (`tuple` of `int`s) : The number of nodes per layer, including
                the input layer and the final layer (number of features). If the model
                is a convolutional neural network, the number of input nodes must be
                equal to the flattened output of the convolutional layers.
            `channels` (`tuple` of `int`s, optional) : The number of applied
                convolutions in each convolutional layer (i.e., the number of resulting
                channels after each set of convolutions).
            `kernel_sizes` (`tuple` of `int`s, optional) : The size of the square
                kernels used in each convolutional layer.
            `pool_sizes` (`tuple` of `int`s, optional) : The size of the square pooling
                kernels used in each max-pooling layer.
        """
        logger = logging.getLogger(__name__)

        # Load configuration file and assign to config_dictionary
        if isinstance(config_filename, str):
            config_filename = Path(config_filename)
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
