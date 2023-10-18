"""This module defines three configuration classes: RunConfig, AdamConfig, and
GeneratorConfig, which are used to configure the machine learning run settings,
the Adam optimizer settings, and the settings for the SurfaceGenerator class.
Each one has a respective `get*ConfigFromFile()` function to easily create a
config object from a YAML or JSON file (see examples directory).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Literal, NamedTuple

import yaml


Parameter = Literal["Bg", "Bth"]
"""Selects either 'Bg' (good solvent parameter) or 'Bth' (thermal blob parameter).
"""


def getDictFromFile(filepath: str | Path) -> dict[str]:
    """Reads a YAML or JSON file and returns the contents as a dictionary.

    Raises a `ValueError` if the extension is incorrect.

    :param filepath: Path to a YAML or JSON file
    :type filepath: str | Path
    :raises ValueError: If the filename extension is not one of '.json',
        '.yaml', or '.yml'
    :return: Nested dictionary based on the contents of the file
    :rtype: dict[str, Any]
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


# @define(kw_only=True)
class RunConfig(NamedTuple):
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

    def keys(self):
        return self._fields
    
    def __getitem__(self, key: str):
        return getattr(self, key)


class AdamConfig(NamedTuple):
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def keys(self):
        return self._fields
    
    def __getitem__(self, key: str):
        return getattr(self, key)


# TODO: Include stripping/trimming
class GeneratorConfig(NamedTuple):
    parameter: Parameter

    batch_size: int

    phi_range: Range
    nw_range: Range
    visc_range: Range

    bg_range: Range
    bth_range: Range
    pe_range: Range

    def keys(self):
        return self._fields
    
    def __getitem__(self, key: str):
        return getattr(self, key)


def getGeneratorConfig(config_dict: dict[str]) -> GeneratorConfig:
    """Convenience class for creating a GeneratorConfig from a nested dictionary
    of settings.

    :param config_dict: A nested dictionary with keys of type string
    :type config_dict: dict[str]
    :return: An instance of :class:`GeneratorConfig` based on the given dictionary
        of settings
    :rtype: GeneratorConfig
    """
    parameter = config_dict["parameter"]
    if parameter not in ("Bg", "Bth"):
        raise ValueError("GeneratorConfig.parameter must be either 'Bg' or 'Bth'.")
    
    batch_size = int(config_dict["batch_size"])

    phi_range = Range(**(config_dict.pop("phi_range")))
    nw_range = Range(**(config_dict.pop("nw_range")))
    visc_range = Range(**(config_dict.pop("visc_range")))
    bg_range = Range(**(config_dict.pop("bg_range")))
    bth_range = Range(**(config_dict.pop("bth_range")))
    pe_range = Range(**(config_dict.pop("pe_range")))
    return GeneratorConfig(
        parameter=parameter,
        batch_size=batch_size,
        phi_range=phi_range,
        nw_range=nw_range,
        visc_range=visc_range,
        bg_range=bg_range,
        bth_range=bth_range,
        pe_range=pe_range,
    )


class Config(NamedTuple):
    """A NamedTuple with parameters `run_config`, `adam_config`, and
    `generator_config`, each of type :class:`RunConfig`, :class:`AdamConfig`,
    and :class:`GeneratorConfig`, respectively.
    """

    run_config: RunConfig
    adam_config: AdamConfig
    generator_config: GeneratorConfig


def loadConfig(filename: str | Path) -> Config:
    """Get configuration settings from a YAML or JSON file (see examples).

    :param filename: Path to a YAML or JSON file from which to read the configurtion
    :type filename: str | Path
    :return: Instance of :class:`Config`
    :rtype: Config
    """
    config_dict = getDictFromFile(filename)

    run_dict = config_dict.get("run")
    run_config = RunConfig(**run_dict)

    adam_dict = config_dict.get("adam")
    adam_config = AdamConfig(**adam_dict)

    generator_dict = config_dict.get("generator")
    generator_config = getGeneratorConfig(generator_dict)

    return Config(run_config, adam_config, generator_config)
