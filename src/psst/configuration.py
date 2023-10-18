"""Configuration classes for the training cycle, Adam optimizer, and SampleGenerator.

Defines three configuration classes: :class:`RunConfig`, :class:`AdamConfig`, and
:class:`GeneratorConfig`, which are used to configure the machine learning run
settings, the Adam optimizer settings, and the settings for the SurfaceGenerator class.
The getConfig function reads a YAML or JSON file and returns a :class:`Config` object,
a NamedTuple of the three config classes (see examples directory).
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Literal, NamedTuple, Optional

from ruamel.yaml import YAML


Parameter = Literal["Bg", "Bth"]
"""Represents either the good solvent parameter ('Bg') or the thermal blob parameter
('Bth').
"""


def getDictFromFile(filepath: str | Path) -> dict[str]:
    """Reads a YAML or JSON file and returns the contents as a dictionary.

    Args:
        filepath (str | Path): The YAML or JSON file to interpret.

    Raises:
        ValueError: If the extension in the filename is not one of ".yaml", ".yml", or
          ".json".

    Returns:
        dict[str]: The contents of the file in dictionary form.
    """
    log = logging.getLogger("psst.main")

    if isinstance(filepath, str):
        filepath = Path(filepath)

    ext = filepath.suffix
    if ext == ".json":
        load = json.load
    elif ext in [".yaml", ".yml"]:
        yaml = YAML(typ="safe", pure=True)
        load = yaml.load
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
    for logarithmic spacing. For use in, e.g., `torch.linspace`, `torch.logspace`.

    Usage:
      >>> Range(1.0, 1e6, num = 100, log_scale = True)

    Attributes:
        min (float): Minimum value of the range.
        max (float): Maximum value of the range.
        num (int): Number of values in the range, including endpoints, default is 0.
        log_scale (bool): If False (the default), the :code:`num` values are evenly
          spaced between :code:`min` and :code:`max`. If True, the values are spaced
          geometrically, such that the respecitve quotients of any two pairs of
          adjacent elements are equal.
    """
    min: float
    max: float
    num: int = 0
    log_scale: bool = False


# @define(kw_only=True)
class RunConfig(NamedTuple):
    """Configuration settings for the training/testing cycle.

    Attributes:
        num_epochs (int): Number of epochs to run
        num_samples_train (int): Number of samples to run through model training per epoch
        num_samples_test (int): Number of samples to run through model testing/validation
          per epoch
        checkpoint_frequency (int): Frequency with which to save the model and optimizer.
          Positive values are number of epochs. Negative values indicate to save when
          the value of the loss function hits a new minimum. Defaults to 0, never saving.
        checkpoint_filename (str): Name of checkpoint file into which to save the model
          and optimizer. Defaults to `"chk.pt"`.
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
    """Configuration settings for the Adam optimizer. See torch.optim.Adam
    documentation for details.
    """
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    foreach: Optional[bool] = None
    maximize: bool = False
    capturable: bool = False
    differentiable: bool = False
    fused: Optional[bool] = None

    def keys(self):
        return self._fields
    
    def __getitem__(self, key: str):
        return getattr(self, key)


# TODO: Include stripping/trimming
class GeneratorConfig(NamedTuple):
    """Configuration settings for the :class:`SampleGenerator` class.

    Attributes:
        parameter (psst.Parameter): Either 'Bg' or 'Bth' to generate viscosity samples
          for the good solvent parameter or the thermal blob parameter, respectively.
        batch_size (int): Number of samples generated per batch.
        phi_range (:class:`psst.Range`): The range of values for the normalized
          concentration :math:`cl^3`.
        nw_range (:class:`psst.Range`): The range of values for the weight-average
          degree of polymerization of the polymers.
        visc_range (:class:`psst.Range`): The range of values for the specific
          viscosity. This is only used for normalization, so `num=0` is fine.
        bg_range (:class:`psst.Range`): The range of values for the good solvent
          parameter. This is only used for normalization, so `num=0` is fine.
        bth_range (:class:`psst.Range`): The range of values for the thermal blob
          parameter. This is only used for normalization, so `num=0` is fine.
        bg_range (:class:`psst.Range`): The range of values for the entanglement
          packing number. This is only used for normalization, so `num=0` is fine.
    """
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

    Args:
        config_dict (dict[str]): A nested dictionary with keys of type string

    Raises:
        ValueError: If :code:`config_dict['parameter']` is not either 'Bg' or 'Bth'.

    Returns:
        :class:`GeneratorConfig`: An instance of :class:`GeneratorConfig` based on the
          given dictionary of settings.
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
    `generator_config`, of types :class:`RunConfig`, :class:`AdamConfig`,
    and :class:`GeneratorConfig`, respectively.
    """

    run_config: RunConfig
    adam_config: AdamConfig
    generator_config: GeneratorConfig


def loadConfig(filename: str | Path) -> Config:
    """Get configuration settings from a YAML or JSON file (see examples).

    Args:
        filename (str | Path): Path to a YAML or JSON file.

    Returns:
        :class:`Config`: _description_
    """
    config_dict = getDictFromFile(filename)

    run_dict = config_dict.get("run")
    run_config = RunConfig(**run_dict)

    adam_dict = config_dict.get("adam", dict())
    adam_config = AdamConfig(**adam_dict)

    generator_dict = config_dict.get("generator")
    generator_config = getGeneratorConfig(generator_dict)

    return Config(run_config, adam_config, generator_config)
