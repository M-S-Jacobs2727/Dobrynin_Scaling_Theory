"""Sample configurations that define the model, training, and testing are defined in
YAML or JSON files in the configurations directory. One of these is passed as the first
command line argument into the main module. This module defines `NNConfig`, the
configuration class, which is initialized from the given configuration file.

TODO: move Resolution to Range? If min and max are attributes of, e.g., nw_range,
why not num_points?
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

    ## Attributes

    `device` (`torch.device`) : The compute device on which all calculations take
        place. If not specified, the CPU is used.
    `output_directory` (`pathlib.Path`) : The directory in which all output will be
        stored. If not specified, the current working directory is used.

    ### Generator parameters
    `mode` (`data_processing.Mode`) : Selects the features that will be generated
        and trained. Can be set as 'good' (only $B_g$ and $P_e$), 'theta' (only
        $B_{th}$ and $P_e$), or 'mixed' (all three).
    `num_nw_strips` (`int`) : If set, the generated surfaces are stripped down such that
        only this many rows in the $N_w$ dimension have data, to emulate experimental
        data. Default: 0.
    `resolution` (`data_processing.Resolution`) : The resolution of the generated
        surfaces, given as a length 2 or length 3 list of ints in the config file. The
        optional third element denotes the resolution of the $\\eta_{sp}$ dimension for
        voxel-based surface images.

    ### Optimizer parameters
    `learning_rate` (`float`) : The learning rate of the PyTorch optimizer Adam.

    ### Range parameters
    `*_range` (`data_processing.Range`) : These objects define the minimum
        (`.min`), maximum (`.max`), and distribution (`.alpha` and `.beta` for the
        Beta distribution, `.mu` and `.sigma` for the LogNormal distribution) of
        their respective parameters.
    `phi_range` : The range of concentrations.
    `nw_range` : The range of weight-average degrees of polymerization.
    `eta_sp_range` : The range of specific viscosity.
    `bg_range` : The range and distribution settings for the blob parameter $B_g$.
    `bth_range` : The range and distribution settings for the blob parameter $B_{th}$.
    `pe_range` : The range and distribution settings for the packing number $P_e$.

    ### Training parameters
    `batch_size` (`int`) : The number of samples given to the model per batch.
    `train_size` (`int`) : The number of samples given to the model over one
        training iteration.
    `test_size` (`int`) : The number of samples given to the model over one
        testing iteration.
    `epochs` (`int`) : The number of training/testing iterations.

    ### Model parameters
    `layer_sizes` (`tuple` of `int`s) : The number of nodes per fully connected layer,
        including the input layer and the final layer (number of features). If the model
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
    num_nw_strips: int
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
                raise SyntaxError(
                    f"Invalid file extension for config file: {extension}\n"
                    "Please use .yaml or .json."
                )

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

        # Surface generation mode
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
            raise SyntaxError(f"Invalid `mode` setting in config file: {mode}")

        # Whether to strip values of Nw from the generated surfaces, for training data
        # more similar in shape to experimental data.
        num_nw_strips = config_dict.get("num_nw_strips")
        if num_nw_strips:
            if int(num_nw_strips) != num_nw_strips:
                raise ValueError(
                    "Config parameter 'num_nw_strips' must be an integer."
                    f" Value: {num_nw_strips}"
                )
            if num_nw_strips < 0:
                raise ValueError(
                    "Config parameter 'num_nw_strips' must be >= 0."
                    f" Value: {num_nw_strips}"
                )
            self.num_nw_strips = int(num_nw_strips)
        else:
            self.num_nw_strips = 0

        # Optimizer learning rate
        self.learning_rate = float(config_dict["learning_rate"])
        if self.learning_rate <= 0:
            raise ValueError(
                "Config parameter 'learning_rate' must be > 0."
                f" Value: {self.learning_rate}"
            )
        logger.debug(f"Loaded {self.learning_rate = :.5f}.")

        # Surface resolution
        self.resolution = Resolution(*config_dict["resolution"])
        if self.resolution.phi <= 0 or self.resolution.Nw <= 0:
            raise ValueError(
                "First two elements of config parameter 'resolution' must be > 0."
                f" Values: ({self.resolution.phi=}, {self.resolution.Nw=})"
            )
        if self.resolution.eta_sp < 0:
            raise ValueError(
                "Third element of config parameter 'resolution' must be >= 0."
                f" Value: {self.resolution.eta_sp}"
            )
        if (
            self.resolution.phi != int(self.resolution.phi)
            or self.resolution.Nw != int(self.resolution.Nw)
            or self.resolution.eta_sp != int(self.resolution.eta_sp)
        ):
            raise ValueError(
                "All elements of config parameter 'resolution must be integers."
                f" Values: ({self.resolution.phi=}, {self.resolution.Nw=},"
                f" {self.resolution.eta_sp})"
            )
        logger.debug(f"Loaded {self.resolution = }.")

        # Min and max values of concentration, Nw, and viscosity
        self.phi_range = Range(
            config_dict["phi_range"]["min"], config_dict["phi_range"]["max"]
        )
        if self.phi_range.min <= 0 or self.phi_range.max <= 0:
            raise ValueError(
                "Min and max values of config parameter 'phi_range' must be > 0."
                f" Values: ({self.phi_range.min=}, {self.phi_range.max=})"
            )
        if self.phi_range.min >= self.phi_range.max:
            raise ValueError(
                "Min value of config parameter 'phi_range' should be less than max"
                f" value. Values: ({self.phi_range.min=}, {self.phi_range.max=})"
            )
        self.nw_range = Range(
            config_dict["nw_range"]["min"], config_dict["nw_range"]["max"]
        )
        if self.nw_range.min <= 0 or self.nw_range.max <= 0:
            raise ValueError(
                "Min and max values of config parameter 'nw_range' must be > 0."
                f" Values: ({self.nw_range.min=}, {self.nw_range.max=})"
            )
        if self.nw_range.min >= self.nw_range.max:
            raise ValueError(
                "Min value of config parameter 'nw_range' should be less than max"
                f" value. Values: ({self.nw_range.min=}, {self.nw_range.max=})"
            )
        self.eta_sp_range = Range(
            config_dict["eta_sp_range"]["min"],
            config_dict["eta_sp_range"]["max"],
        )
        if self.eta_sp_range.min <= 0 or self.eta_sp_range.max <= 0:
            raise ValueError(
                "Min and max values of config parameter 'eta_sp_range' must be > 0."
                f" Values: ({self.eta_sp_range.min=}, {self.eta_sp_range.max=})"
            )
        if self.eta_sp_range.min >= self.eta_sp_range.max:
            raise ValueError(
                "Min value of config parameter 'eta_sp_range' should be less than max"
                f" value. Values: ({self.eta_sp_range.min=}, {self.eta_sp_range.max=})"
            )
        logger.debug(
            f"Loaded {self.phi_range = }, {self.nw_range = }, {self.eta_sp_range = }."
        )

        # Min, max, and distribution definitions for Bg, Bth, and Pe
        # For the distributions, we use .get() instead of [] because it returns None on
        # an invalid key. For min and max, we require these to be specified, so they
        # will raise an error otherwise.
        self.bg_range = Range(
            min=config_dict["bg_range"]["min"],
            max=config_dict["bg_range"]["max"],
            alpha=config_dict["bg_range"].get("alpha"),
            beta=config_dict["bg_range"].get("beta"),
            mu=config_dict["bg_range"].get("mu"),
            sigma=config_dict["bg_range"].get("sigma"),
        )
        if self.bg_range.min <= 0 or self.bg_range.max <= 0:
            raise ValueError(
                "Min and max values of config parameter 'bg_range' must be > 0."
                f" Values: ({self.bg_range.min=}, {self.bg_range.max=})"
            )
        if self.bg_range.min >= self.bg_range.max:
            raise ValueError(
                "Min value of config parameter 'bg_range' should be less than max"
                f" value. Values: ({self.bg_range.min=}, {self.bg_range.max=})"
            )
        self.bth_range = Range(
            min=config_dict["bth_range"]["min"],
            max=config_dict["bth_range"]["max"],
            alpha=config_dict["bth_range"].get("alpha"),
            beta=config_dict["bth_range"].get("beta"),
            mu=config_dict["bth_range"].get("mu"),
            sigma=config_dict["bth_range"].get("sigma"),
        )
        if self.bth_range.min <= 0 or self.bth_range.max <= 0:
            raise ValueError(
                "Min and max values of config parameter 'bth_range' must be > 0."
                f" Values: ({self.bth_range.min=}, {self.bth_range.max=})"
            )
        if self.bth_range.min >= 1 or self.bth_range.max >= 1:
            raise ValueError(
                "Min and max values of config parameter 'bth_range' must be < 1."
                f" Values: ({self.bth_range.min=}, {self.bth_range.max=})"
            )
        if self.bth_range.min >= self.bth_range.max:
            raise ValueError(
                "Min value of config parameter 'bth_range' should be less than max"
                f" value. Values: ({self.bth_range.min=}, {self.bth_range.max=})"
            )
        self.pe_range = Range(
            min=config_dict["pe_range"]["min"],
            max=config_dict["pe_range"]["max"],
            alpha=config_dict["pe_range"].get("alpha"),
            beta=config_dict["pe_range"].get("beta"),
            mu=config_dict["pe_range"].get("mu"),
            sigma=config_dict["pe_range"].get("sigma"),
        )
        if self.pe_range.min <= 0 or self.pe_range.max <= 0:
            raise ValueError(
                "Min and max values of config parameter 'pe_range' must be > 0."
                f" Values: ({self.pe_range.min=}, {self.pe_range.max=})"
            )
        if self.pe_range.min >= self.pe_range.max:
            raise ValueError(
                "Min value of config parameter 'pe_range' should be less than max"
                f" value. Values: ({self.pe_range.min=}, {self.pe_range.max=})"
            )

        # Check for incorrect combinations of (mu, sigma) and (alpha, beta) pairs
        for param_range in [self.bg_range, self.bth_range, self.pe_range]:
            if (
                param_range.mu
                and param_range.sigma
                and param_range.alpha
                and param_range.beta
            ):
                raise SyntaxError(
                    "Only one pair of (alpha, beta) or (mu, sigma) may be specified."
                    " Instead, both are specified."
                )
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
        self.batch_size = config_dict["batch_size"]
        if int(self.batch_size) != self.batch_size:
            raise ValueError(
                "Config parameter 'batch_size' must be an integer."
                f" Value: {self.batch_size}"
            )
        if self.batch_size < 0:
            raise ValueError(
                "Config parameter 'batch_size' must be >= 0."
                f" Value: {self.batch_size}"
            )
        self.train_size = config_dict["train_size"]
        if int(self.train_size) != self.train_size:
            raise ValueError(
                "Config parameter 'train_size' must be an integer."
                f" Value: {self.train_size}"
            )
        if self.train_size < 0:
            raise ValueError(
                "Config parameter 'train_size' must be >= 0."
                f" Value: {self.train_size}"
            )
        self.test_size = config_dict["test_size"]
        if int(self.test_size) != self.test_size:
            raise ValueError(
                "Config parameter 'test_size' must be an integer."
                f" Value: {self.test_size}"
            )
        if self.test_size < 0:
            raise ValueError(
                "Config parameter 'test_size' must be >= 0." f" Value: {self.test_size}"
            )
        self.epochs = config_dict["epochs"]
        if int(self.epochs) != self.epochs:
            raise ValueError(
                "Config parameter 'epochs' must be an integer." f" Value: {self.epochs}"
            )
        if self.epochs < 0:
            raise ValueError(
                "Config parameter 'epochs' must be >= 0." f" Value: {self.epochs}"
            )
        logger.debug(
            f"Loaded {self.batch_size = }, {self.train_size = }, {self.test_size = },"
            f" and {self.epochs = }."
        )

        # Number of nodes in each linear NN layer
        self.layer_sizes = tuple(config_dict["layer_sizes"])
        if self.layer_sizes[-1] != 3 and self.mode is Mode.MIXED:
            raise SyntaxError(
                "For the 'mixed' mode, the final element in `layer_sizes` must be 3."
            )
        if self.layer_sizes[-1] != 2 and (
            self.mode is Mode.GOOD or self.mode is Mode.THETA
        ):
            raise SyntaxError(
                f"For the '{self.mode}' mode, the final element in `layer_sizes` must"
                " be 2."
            )
        for size in self.layer_sizes:
            if int(size) != size:
                raise ValueError(
                    "Config parameter 'layer_sizes' must be a list of integers."
                    f" Values: {self.layer_sizes}"
                )
            if size <= 0:
                raise ValueError(
                    "Config parameter 'layer_sizes' must be > 0."
                    f" Values: {self.layer_sizes}"
                )
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
                raise SyntaxError(
                    "Expected configuration parameters `channels`, `kernel_sizes`, and"
                    "`pool_sizes` to have equal length. Instead, the lengths are"
                    f" {len(self.channels)}, {len(self.kernel_sizes)}, and"
                    f" {len(self.pool_sizes)}, respectively."
                )

            for size in self.channels:
                if int(size) != size:
                    raise ValueError(
                        "Config parameter 'channels' must be a list of integers."
                        f" Values: {self.channels}"
                    )
                if size <= 0:
                    raise ValueError(
                        "Config parameter 'channels' must be > 0."
                        f" Values: {self.channels}"
                    )
            for size in self.kernel_sizes:
                if int(size) != size:
                    raise ValueError(
                        "Config parameter 'kernel_sizes' must be a list of integers."
                        f" Values: {self.kernel_sizes}"
                    )
                if size <= 0:
                    raise ValueError(
                        "Config parameter 'kernel_sizes' must be > 0."
                        f" Values: {self.kernel_sizes}"
                    )
            for size in self.pool_sizes:
                if int(size) != size:
                    raise ValueError(
                        "Config parameter 'pool_sizes' must be a list of integers."
                        f" Values: {self.pool_sizes}"
                    )
                if size <= 0:
                    raise ValueError(
                        "Config parameter 'pool_sizes' must be > 0."
                        f" Values: {self.pool_sizes}"
                    )
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
