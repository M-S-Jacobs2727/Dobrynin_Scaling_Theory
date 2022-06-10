from dataclasses import dataclass, field
from typing import List, NamedTuple, Tuple

import numpy as np
import torch
import torch.distributions

SHIFT = 1e-4


class Resolution(NamedTuple):
    phi: int
    Nw: int
    eta_sp: int = 0


class Param(NamedTuple):
    min: float
    max: float
    mu: float = 0
    sigma: float = 0
    alpha: float = 0
    beta: float = 0


@dataclass
class Config:
    device: torch.device
    phi_param: Param = Param(1e-6, 1e-2)
    nw_param: Param = Param(100, 1e5)
    eta_sp_param: Param = Param(1, 1e6)
    bg_param: Param = Param(0.3, 1.1)
    bth_param: Param = Param(0.2, 0.8)
    pe_param: Param = Param(5, 20)
    batch_size: int = 100
    train_size: int = 700000
    test_size: int = 2000
    epochs: int = 100
    resolution: Resolution = Resolution(128, 128)
    lr: float = 1e-3


@dataclass
class NNConfig(Config):
    layers: List[int] = field(default=[1024, 1024])


@dataclass
class CNNConfig(NNConfig):
    channels: List[int] = field(default=[4, 16, 64])
    kernels: List[int] = field(default=[5, 5, 5])
    pools: List[int] = field(default=[2, 2, 2])


def param_dist(param: Param) -> torch.distributions.Distribution:
    if param.alpha and param.beta:
        return torch.distributions.Beta(param.alpha, param.beta)
    if param.mu and param.sigma:
        return torch.distributions.LogNormal(param.mu, param.sigma)
    return torch.distributions.Uniform(param.min, param.max)


def get_Bth_from_Bg(Bg: torch.Tensor) -> torch.Tensor:
    Bth = 0.54 * Bg + 0.05
    Bth += Bth * 0.05 * torch.normal(torch.zeros_like(Bth), torch.ones_like(Bth))
    return Bth


def normalize_params(
    Bg: torch.Tensor,
    Bth: torch.Tensor,
    Pe: torch.Tensor,
    BG: Param,
    BTH: Param,
    PE: Param,
) -> Tuple[torch.Tensor, ...]:
    """Simple linear normalization."""
    Bg = (Bg - BG.min - SHIFT) / (BG.max - BG.min)
    Bth = (Bth - BTH.min - SHIFT) / (BTH.max - BTH.min)
    Pe = (Pe - PE.min - SHIFT) / (PE.max - PE.min)
    return Bg, Bth, Pe


def unnormalize_params(
    Bg: torch.Tensor,
    Bth: torch.Tensor,
    Pe: torch.Tensor,
    BG: Param,
    BTH: Param,
    PE: Param,
) -> Tuple[torch.Tensor, ...]:
    """Simple linear normalization."""
    Bg = Bg * (BG.max - BG.min) + BG.min + SHIFT
    Bth = Bth * (BTH.max - BTH.min) + BTH.min + SHIFT
    Pe = Pe * (PE.max - PE.min) + PE.min + SHIFT
    return Bg, Bth, Pe


def preprocess_visc(eta_sp: torch.Tensor, ETA_SP: Param) -> torch.Tensor:
    """Add noise, cap the values, take the log, then normalize."""
    eta_sp += (
        eta_sp * 0.05 * torch.normal(torch.zeros_like(eta_sp), torch.ones_like(eta_sp))
    )
    eta_sp = torch.fmin(eta_sp, torch.tensor(ETA_SP.max))
    eta_sp = torch.fmax(eta_sp, torch.tensor(ETA_SP.min))
    return normalize_visc(eta_sp, ETA_SP)


def normalize_visc(eta_sp: torch.Tensor, ETA_SP: Param) -> torch.Tensor:
    return torch.log(eta_sp / ETA_SP.min) / np.log(ETA_SP.max / ETA_SP.min)
