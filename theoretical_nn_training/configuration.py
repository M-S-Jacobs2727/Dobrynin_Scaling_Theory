"""Sample configurations that define the model, training, and testing are defined in
YAML or JSON files in the configurations directory. One of these is passed as the first
command line argument into the main module. This module defines `NNConfig`, the
configuration class, which is initialized from the given configuration file.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import yaml

from theoretical_nn_training.data_processing import Mode, Range, Resolution


@dataclass()
class NNConfig:
    """Configuration dataclass. Reads a YAML or JSON configuration file and sets
    corresponding attributes for the returned object. The attributes listed below
    are read from the configuration file with the same hierarchical structure.

    Attributes:

    `device` (`torch.device`) : The compute device on which all calculations take
        place. If not specified, the CPU is used.
    `output_directory` (`pathlib.Path`) : The directory in which all output will be
        stored. If not specified, the current working directory is used.
    `mode` (`data_processing.Mode`) : Selects the features that will be generated
        and trained. Can be set as 'good' (only $B_g$ and $P_e$), 'theta' (only
        $B_{th}$ and $P_e$), or 'mixed' (all three).

    `learning_rate` (`float`) : The learning rate of the PyTorch optimizer Adam.

    `*_range` (`data_processing.Range`) : These objects define the minimum
        (`.min`), maximum (`.max`), and distribution (`.alpha` and `.beta` for the
        Beta distribution, `.mu` and `.sigma` for the LogNormal distribution) of
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

    Note: If the model is a convolutional neural network, then each of `channels`,
    `kernel_sizes`, and `pool_sizes` must be specified and have equal lengths.
    The architecture of the convolutional layers is a sequence of (convolution,
    ReLU activation layer, max-pool) sequences, the result of which is fed into
    a linear neural network.
    """

    device: torch.device
    output_directory: Path
    mode: Mode
    learning_rate: float
    resolution: Resolution
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

    def __init__(self, config_filename: Union[Path, str]) -> None:
        logger = logging.getLogger("__main__")

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
                logger.exception(f"Invalid file extension: {extension}")
                raise

        # PyTorch device
        if config_dict.get("device") is None or config_dict.get("device") == "cpu":
            self.device = torch.device("cpu")
        elif config_dict["device"] == "cuda" and not torch.cuda.is_available():
            logger.warn("Warning: No CUDA-enabled devices found. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config_dict["device"])
        logger.info(f"Set device to {self.device.type}")

        # Output directory
        if out_dir := config_dict.get("output_directory"):
            self.output_directory = Path(out_dir)
            if not self.output_directory.is_dir():
                logger.warn(
                    f"Warning: Output directory {self.output_directory.absolute()} not"
                    "  found. Creating it now."
                )
                self.output_directory.mkdir(parents=True)
        else:
            self.output_directory = Path(".")
        logger.debug(f"Set output directory to {self.output_directory.absolute()}")

        # Generator and learning mode
        mode = config_dict.get("mode")
        if mode is None:
            self.mode = Mode.MIXED
            logger.warn(
                "Warning: configuration setting `mode` not set. Setting to 'mixed'."
            )
        elif mode == "mixed":
            self.mode = Mode.MIXED
        elif mode == "good":
            self.mode = Mode.GOOD
        elif mode == "theta":
            self.mode = Mode.THETA
        else:
            logger.exception(f"Invalid `mode` setting: {mode}")
            raise

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
        # For the distributions, we use .get() instead of [] because it returns None on
        # an invalid key. For min and max, we require these to be specified, so they
        # will raise an error otherwise.
        self.bg_range = Range(
            config_dict["bg_range"]["min"],
            config_dict["bg_range"]["max"],
            alpha=config_dict["bg_range"].get("alpha"),
            beta=config_dict["bg_range"].get("beta"),
            mu=config_dict["bg_range"].get("mu"),
            sigma=config_dict["bg_range"].get("sigma"),
        )
        self.bth_range = Range(
            config_dict["bth_range"]["min"],
            config_dict["bth_range"]["max"],
            alpha=config_dict["bth_range"].get("alpha"),
            beta=config_dict["bth_range"].get("beta"),
            mu=config_dict["bth_range"].get("mu"),
            sigma=config_dict["bth_range"].get("sigma"),
        )
        self.pe_range = Range(
            config_dict["pe_range"]["min"],
            config_dict["pe_range"]["max"],
            alpha=config_dict["pe_range"].get("alpha"),
            beta=config_dict["pe_range"].get("beta"),
            mu=config_dict["pe_range"].get("mu"),
            sigma=config_dict["pe_range"].get("sigma"),
        )

        # Check for incorrect combinations of (mu, sigma) and (alpha, beta) pairs
        for param_range in [self.bg_range, self.bth_range, self.pe_range]:
            if (
                param_range.mu
                and param_range.sigma
                and param_range.alpha
                and param_range.beta
            ):
                logger.exception(
                    "Only one pair of (alpha, beta) or (mu, sigma) may be specified."
                    " Instead, both are specified."
                )
                raise
            if (param_range.alpha is None) ^ (param_range.beta is None):
                logger.warn("Only one of alpha/beta is specified. Ignoring.")
                param_range.alpha = None
                param_range.beta = None
            if (param_range.mu is None) ^ (param_range.sigma is None):
                logger.warn("Only one of mu/sigma is specified. Ignoring.")
                param_range.mu = None
                param_range.sigma = None
        logger.debug(
            f"Loaded {self.bg_range = }, {self.bth_range = }, {self.pe_range = }."
        )

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

        # Number of nodes in each linear NN layer
        self.layer_sizes = tuple(config_dict["layer_sizes"])
        if self.layer_sizes[-1] == 2 and self.mode is Mode.MIXED:
            logger.exception(
                "For the 'mixed' mode, the final element in `layer_sizes` must be 3."
            )
            raise
        if self.layer_sizes[-1] == 3 and (
            self.mode is Mode.GOOD or self.mode is Mode.THETA
        ):
            logger.exception(
                f"For the '{self.mode}' mode, the final element in `layer_sizes` must"
                " be 2."
            )
            raise
        logger.debug(f"Loaded {self.layer_sizes = }.")

        # Define these three hyperparameters in the configuration file with equal
        # lengths to specify a convolutional neural network.
        channels = config_dict.get("channels")
        kernel_sizes = config_dict.get("kernel_sizes")
        pool_sizes = config_dict.get("pool_sizes")
        if channels and kernel_sizes and pool_sizes:
            self.channels = tuple(channels)
            self.kernel_sizes = tuple(kernel_sizes)
            self.pool_sizes = tuple(pool_sizes)
            if not (
                len(self.channels) == len(self.kernel_sizes) == len(self.pool_sizes)
            ):
                logger.exception(
                    "Expected configuration parameters `channels`, `kernel_sizes`, and"
                    "`pool_sizes` to have equal length. Instead, the lengths are"
                    f" {len(self.channels)}, {len(self.kernel_sizes)}, and"
                    f" {len(self.pool_sizes)}, respectively."
                )
                raise
            logger.debug(
                f"Loaded {self.channels = }, {self.kernel_sizes = }, and"
                f" {self.pool_sizes = }."
            )
        elif channels or kernel_sizes or pool_sizes:
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
