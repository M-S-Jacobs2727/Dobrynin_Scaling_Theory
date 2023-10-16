"""This module defines three configuration classes: RunConfig, AdamConfig, and
GeneratorConfig, which are used to configure the machine learning run settings,
the Adam optimizer settings, and the settings for the SurfaceGenerator class.
Each one has a respective `get*ConfigFromFile()` function to easily create a
config object from a YAML or JSON file (see examples directory).
"""
import json
import logging
from pathlib import Path
from typing import Literal, NamedTuple

from attrs import define, field, asdict
import yaml


Parameter = Literal["Bg", "Bth"]
"""Selects either 'Bg' (good solvent parameter) or 'Bth' (thermal blob parameter).
"""


def getDictFromFile(filepath: str | Path):
    """Reads a YAML or JSON file and returns the contents as a dictionary.

    Raises a `ValueError` if the extension is incorrect.
    """
    log = logging.getLogger("psst.main")

    if isinstance(filepath, str):
        filepath = Path(filepath)

    ext = filepath.suffix
    if ext == ".json":
        load = json.load
    elif ext in [".yaml", ".yml"]:
        load = yaml.safe_load
    else:
        raise ValueError(
            f"Invalid file extension for config file: {ext}\n"
            "Please use .yaml or .json."
        )

    log.info("Loading configuration from %s", str(filepath))

    with open(filepath, "r") as f:
        config_dict = dict(load(f))
        log.debug("Loaded configuration: %s", str(config_dict))

    return config_dict


def dictToRange(d: dict[str]):
    """Simple converter for the GeneratorConfig attrs class."""
    return Range(**d)


class Range(NamedTuple):
    """Specifies a range of values between :code:`min` and :code:`max`,
    optionally specifying :code:`num` for number of points and :code:`log_scale`
    for logarithmic spacing. For use in, e.g., `torch.linspace`,
    `torch.logspace`.

    Usage:
      >>> Range(1.0, 1e6, num = 100, log_scale = True)
    """

    min: float
    max: float
    num: int = 0
    log_scale: bool = False


@define(kw_only=True)
class RunConfig:
    """Configuration class for training/testing run settings.

    Fields (keyword-only):

    :param train_size: Number of samples to run through model training per epoch
    :type train_size: int
    :param test_size: Number of samples to run through model testing/validation per epoch
    :type test_size: int
    :param num_epochs: Number of epochs to run
    :type num_epochs: int
    :param read_checkpoint_file: Name of checkpoint file from which to read the model
        and optimizer. Defaults to `None`, starting with a fresh model.
    :type read_checkpoint_file: str, optional
    :param write_checkpoint_file: Name of checkpoint file into which to save the model
        and optimizer. Defaults to `"chk.pt"`.
    :type write_checkpoint_file: str, optional
    """

    train_size: int = field(converter=int)
    test_size: int = field(converter=int)
    num_epochs: int = field(converter=int)
    read_checkpoint_file: str | None = field(default=None)
    write_checkpoint_file: str = field(default="chk.pt")


def getRunConfigFromFile(filename: str | Path):
    """Reads a YAML or JSON file containing run settings for a training/testing
    run and returns a :class:`RunConfig` object.

    :param filename: The name of a YAML or JSON file
    :type filename: Path-like
    :return: The run configuration settings.
    :rtype: :class:`RunConfig`
    """
    config_dict = getDictFromFile(filename)
    return RunConfig(**config_dict)


@define(kw_only=True)
class AdamConfig:
    lr: float = field(converter=float, default=1e-3)
    betas: tuple[float, float] = field(default=(0.7, 0.9))
    eps: float = field(converter=float, default=1e-9)
    weight_decay: float = field(converter=float, default=0.0)

    def asdict(self):
        return asdict(self)


def getAdamConfigFromFile(filename: str | Path):
    config_dict = getDictFromFile(filename)
    return AdamConfig(**config_dict)


# TODO: Include stripping/trimming
@define(kw_only=True)
class GeneratorConfig:
    parameter: Parameter
    batch_size: int = field(default=64, converter=int)

    phi_range: Range = field(default=Range(3e-5, 2e-2, 224), converter=dictToRange)
    nw_range: Range = field(default=Range(100, 1e5, 224), converter=dictToRange)
    visc_range: Range = field(default=Range(1, 1e6), converter=dictToRange)

    bg_range: Range = field(default=Range(0.36, 1.55), converter=dictToRange)
    bth_range: Range = field(default=Range(0.22, 0.82), converter=dictToRange)
    pe_range: Range = field(default=Range(3.2, 13.5), converter=dictToRange)

    def asdict(self) -> dict[str]:
        return asdict(self)


def getGeneratorConfigFromFile(filename: str | Path):
    config_dict = getDictFromFile(filename)
    return GeneratorConfig(**config_dict)
