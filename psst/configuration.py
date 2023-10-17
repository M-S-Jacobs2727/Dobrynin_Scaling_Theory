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
    """
    :param num_epochs: Number of epochs to run
    :type num_epochs: int
    :param num_samples_train: Number of samples to run through model training per epoch
    :type num_samples_train: int
    :param num_samples_test: Number of samples to run through model testing/validation
    per epoch
    :type num_samples_test: int
    :param checkpoint_frequency: Frequency with which to save the model and optimizer.
        Positive values are number of epochs. Negative values indicate to save when
        the value of the loss function hits a new minimum. Defaults to 0, never saving.
    :type checkpoint_frequency: int, optional
    :param checkpoint_filename: Name of checkpoint file into which to save the model
        and optimizer. Defaults to `"chk.pt"`.
    :type checkpoint_filename: str, optional
    """

    num_epochs: int
    num_samples_train: int
    num_samples_test: int
    checkpoint_frequency: int = 0
    checkpoint_filename: str = "chk.pt"


@define(kw_only=True)
class AdamConfig:
    lr: float = field(converter=float, default=1e-3)
    betas: tuple[float, float] = field(default=(0.7, 0.9))
    eps: float = field(converter=float, default=1e-9)
    weight_decay: float = field(converter=float, default=0.0)

    def asdict(self):
        return asdict(self)


# TODO: Include stripping/trimming
@define(kw_only=True)
class GeneratorConfig:
    parameter: Parameter

    batch_size: int = 64

    phi_range: Range
    nw_range: Range
    visc_range: Range

    bg_range: Range
    bth_range: Range
    pe_range: Range

    def asdict(self) -> dict[str]:
        return asdict(self)


def getGeneratorConfig(config_dict: dict[str]) -> GeneratorConfig:
    phi_range = Range(**(config_dict.pop("phi_range")))
    nw_range = Range(**(config_dict.pop("nw_range")))
    visc_range = Range(**(config_dict.pop("visc_range")))
    bg_range = Range(**(config_dict.pop("bg_range")))
    bth_range = Range(**(config_dict.pop("bth_range")))
    pe_range = Range(**(config_dict.pop("pe_range")))
    return GeneratorConfig(
        phi_range=phi_range,
        nw_range=nw_range,
        visc_range=visc_range,
        bg_range=bg_range,
        bth_range=bth_range,
        pe_range=pe_range,
        **config_dict,
    )


@define
class Config:
    run_config: RunConfig
    adam_config: AdamConfig
    generator_config: GeneratorConfig


def getConfig(filename: str | Path) -> Config:
    config_dict = getDictFromFile(filename)

    run_dict = config_dict.get("run", dict())
    run_config = RunConfig(**run_dict)

    adam_dict = config_dict.get("adam", dict())
    adam_config = AdamConfig(**adam_dict)

    generator_dict = config_dict.get("generator", dict())
    generator_config = getGeneratorConfig(generator_dict)

    return Config(run_config, adam_config, generator_config)
