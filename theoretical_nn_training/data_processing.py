from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributions

SHIFT = 1e-4


@dataclass
class Resolution:
    """Resolution of generated surfaces. Dimensions:
    - `phi` (concentration)
    - `Nw` (weight-average degree of polymerization)
    - `eta_sp` (specific viscosity) (optional, for 3D representations only)
    """

    phi: int
    Nw: int
    eta_sp: int = 0


@dataclass
class Range:
    """Defines the minimum and maximum values of a distribution of allowed values for
    parameters. Optionally also allows for the specification of mu and sigma for a
    Normal or LogNormal distribution, and alpha and beta for a Beta distribution. If
    none of these are specified, then a Uniform distribution is assumed.
    """

    min: float
    max: float
    mu: Optional[float] = None
    sigma: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None


def param_dist(param: Range) -> torch.distributions.Distribution:
    """Returns a generator that produces a distribution of values for the parameters
    Bg, Bth, and Pe. These can be the Beta distribution (if param.alpha and param.beta
    are defined), the LogNormal distribution (if param.mu and param.sigma are defined)
    or the Uniform distribution otherwise.
    """
    if param.alpha and param.beta:
        return torch.distributions.Beta(param.alpha, param.beta)
    if param.mu and param.sigma:
        return torch.distributions.LogNormal(param.mu, param.sigma)
    return torch.distributions.Uniform(param.min, param.max)


def get_Bth_from_Bg(Bg: torch.Tensor) -> torch.Tensor:
    """This computes Bth from Bg on the assumption that they have a linear
    relationship.
    """
    Bth = 0.54 * Bg + 0.05
    Bth += Bth * 0.1 * torch.normal(torch.zeros_like(Bth), torch.ones_like(Bth))
    return Bth


def normalize_params(
    Bg: torch.Tensor,
    Bth: torch.Tensor,
    Pe: torch.Tensor,
    bg_range: Range,
    bth_range: Range,
    pe_range: Range,
) -> Tuple[torch.Tensor, ...]:
    """Performs simple linear normalization."""
    Bg = (Bg - bg_range.min - SHIFT) / (bg_range.max - bg_range.min)
    Bth = (Bth - bth_range.min - SHIFT) / (bth_range.max - bth_range.min)
    Pe = (Pe - pe_range.min - SHIFT) / (pe_range.max - pe_range.min)
    return Bg, Bth, Pe


def unnormalize_params(
    Bg: torch.Tensor,
    Bth: torch.Tensor,
    Pe: torch.Tensor,
    bg_range: Range,
    bth_range: Range,
    pe_range: Range,
) -> Tuple[torch.Tensor, ...]:
    """Inverts simple linear normalization."""
    Bg = Bg * (bg_range.max - bg_range.min) + bg_range.min + SHIFT
    Bth = Bth * (bth_range.max - bth_range.min) + bth_range.min + SHIFT
    Pe = Pe * (pe_range.max - pe_range.min) + pe_range.min + SHIFT
    return Bg, Bth, Pe


def unnormalize_visc(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Inverts a simple linear normalization of the specific viscosity."""
    return (
        torch.exp(eta_sp * np.log(eta_sp_range.max / eta_sp_range.min))
        * eta_sp_range.min
    )


def normalize_visc(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Performs a simple linear normalization of the natural log of the specific
    viscosity.
    """
    return torch.log(eta_sp / eta_sp_range.min) / np.log(
        eta_sp_range.max / eta_sp_range.min
    )


def preprocess_visc(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Add noise, cap the values, take the log, then normalize."""
    eta_sp += (
        eta_sp * 0.05 * torch.normal(torch.zeros_like(eta_sp), torch.ones_like(eta_sp))
    )
    eta_sp = torch.fmin(eta_sp, torch.tensor(eta_sp_range.max))
    eta_sp = torch.fmax(eta_sp, torch.tensor(eta_sp_range.min))
    # eta_sp[eta_sp == eta_sp_range.max] = eta_sp_range.min
    return normalize_visc(eta_sp, eta_sp_range)
