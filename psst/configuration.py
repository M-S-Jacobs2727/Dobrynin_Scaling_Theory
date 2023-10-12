"""Sample configurations that define the model, training, and testing are defined in
YAML or JSON files in the configurations directory. One of these is passed as the first
command line argument into the main module. This module defines `NNConfig`, the
configuration class, which is initialized from the given configuration file.
"""
from enum import Enum
import json
import logging
from pathlib import Path
from typing import NamedTuple

from attrs import define, field, asdict
import numpy as np
import torch
import yaml


class ParameterChoice(Enum):
    Bg = "Bg"
    Bth = "Bth"

def convertToRange(d):
    if isinstance(d, np.ndarray):
        return torch.as_tensor(d)
    if isinstance(d, torch.Tensor):
        return d
    if isinstance(d, dict):
        return getRange(**d)
    if isinstance(d, list) or isinstance(d, tuple):
        return getRange(*d)
    raise ValueError("Invalid type %s" % type(d))


def getRange(min: float, max: float, num: int, geom: bool = True) -> torch.Tensor:
    if geom:
        return torch.logspace(np.log10(min), np.log10(max), num, dtype=torch.float32)
    return torch.linspace(min, max, num, dtype=torch.float32)


# class DistType(Enum):
#     uniform = 1
#     normal = 2
#     beta = 3

def getDist(d: list):
    return Dist(*d)

class Dist(NamedTuple):
    min: float
    max: float
    # dist_type: DistType = DistType.uniform

# TODO: Include stripping/trimming
@define(kw_only=True)
class GeneratorConfig:
    parameter: ParameterChoice = field(converter=ParameterChoice)
    batch_size: int = field(default=64, converter=int)

    phi_range: torch.Tensor = field(default=getRange(3e-5, 2e-2, 224), converter=convertToRange)
    nw_range: torch.Tensor = field(default=getRange(100, 1e5, 224), converter=convertToRange)

    eta_sp_dist: Dist = field(default=Dist(1, 1e6), converter=getDist)
    bg_dist: Dist = field(default=Dist(0.36, 1.55), converter=getDist)
    bth_dist: Dist = field(default=Dist(0.22, 0.82), converter=getDist)
    pe_dist: Dist = field(default=Dist(3.2, 13.5), converter=getDist)

    def asdict(self):
        return asdict(self)

@define(kw_only=True)
class AdamConfig:
    lr: float = field(converter=float, default=1e-3)
    betas: tuple[float, float] = field(default=(0.7, 0.9))
    eps: float = field(converter=float, default=1e-9)
    weight_decay: float = field(converter=float, default=0.0)

    def asdict(self):
        return asdict(self)

@define(eq=False, kw_only=True)
class Configuration:
    generator_config: GeneratorConfig
    adam_config: AdamConfig

    checkpoint_file: str
    continuing: bool = field(converter=bool, default=False)
    train_size: int = field(converter=int, default=51200)
    test_size: int = field(converter=int, default=21952)
    epochs: int = field(converter=int, default=300)



def getConfig(filepath: Path | str) -> Configuration:
    log = logging.getLogger("psst.main")

    if isinstance(filepath, str):
        filepath = Path(filepath)

    log.info("Loading configuration from %s", str(filepath))

    with open(filepath, "r") as f:
        extension = filepath.suffix
        if extension in [".yaml", ".yml"]:
            config_dict = dict(yaml.safe_load(f))
        elif extension == ".json":
            config_dict = dict(json.load(f))
        else:
            raise SyntaxError(
                f"Invalid file extension for config file: {extension}\n"
                "Please use .yaml or .json."
            )
        log.debug("Loaded configuration: %s", str(config_dict))

    generator_dict = config_dict.pop("generator", dict())
    adam_dict = config_dict.pop("adam", dict())
    config = Configuration(
        generator_config=GeneratorConfig(**generator_dict),
        adam_config=AdamConfig(**adam_dict),
        **config_dict
    )

    log.debug("Fully initialized configuration: \n%s", config)
    return config
