"""This is somewhat of a catch-all for classes and functions that are used throughout
the codebase. `Resolution` is a dataclass for the resolution of the
generated surfaces, either 2D or 3D; `Range` defines the minimum, maximum, and
distributions for various parameters; and `normalize_params`, `unnormalize_params`,
`preprocess_eta_sp`, etc. perform basic normalization operations on the data and
features.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.distributions


class Mode(Enum):
    """An enum for selecting the features to use and train against.

    Attributes:
        `MIXED`: All features Bg, Bth, and Pe
        `THETA`: Only Bth and Pe
        `GOOD`: Only Bg and Pe
    """

    MIXED = "mixed"
    THETA = "theta"
    GOOD = "good"


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
    features. Optionally also allows for the specification of mu and sigma for a
    Normal or LogNormal distribution, and alpha and beta for a Beta distribution. If
    none of these are specified, then a Uniform distribution is assumed.
    """

    min: float
    max: float
    mu: Optional[float] = None
    sigma: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None


def feature_distribution(
    feature_range: Range, batch_size: int
) -> torch.distributions.Distribution:
    """Returns a generator that can be sampled in batches of size `batch_size` for the
    features Bg, Bth, or Pe. These can be the Beta distribution (if feature_range.alpha
    and feature_range.beta are defined), the LogNormal distribution (if feature_range.mu
    and feature_range.sigma are defined), or the Uniform distribution otherwise.
    """
    if feature_range.alpha and feature_range.beta:
        return torch.distributions.Beta(
            torch.zeros((batch_size,), dtype=torch.float) + feature_range.alpha,
            torch.zeros((batch_size,), dtype=torch.float) + feature_range.beta,
        )
    if feature_range.mu and feature_range.sigma:
        return torch.distributions.LogNormal(
            torch.zeros((batch_size,), dtype=torch.float) + feature_range.mu,
            torch.zeros((batch_size,), dtype=torch.float) + feature_range.sigma,
        )
    return torch.distributions.Uniform(
        torch.zeros((batch_size,), dtype=torch.float) + feature_range.min,
        torch.zeros((batch_size,), dtype=torch.float) + feature_range.max,
    )


def get_Bth_from_Bg(Bg: torch.Tensor) -> torch.Tensor:
    """This computes Bth from Bg on the assumption that they have a linear
    relationship.
    """
    Bth = 0.54 * Bg + 0.05
    Bth += Bth * 0.1 * torch.normal(torch.zeros_like(Bth), torch.ones_like(Bth))
    return Bth


def normalize_feature(
    feature: torch.Tensor,
    feature_range: Range,
) -> torch.Tensor:
    """Performs simple linear normalization."""
    return (feature - feature_range.min) / (feature_range.max - feature_range.min)


def unnormalize_feature(
    feature: torch.Tensor,
    feature_range: Range,
) -> torch.Tensor:
    """Inverts simple linear normalization."""
    return feature * (feature_range.max - feature_range.min) + feature_range.min


def unnormalize_eta_sp(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Inverts a simple linear normalization of the specific viscosity."""
    return (
        torch.exp(eta_sp * np.log(eta_sp_range.max / eta_sp_range.min))
        * eta_sp_range.min
    )


def normalize_eta_sp(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Performs a simple linear normalization of the natural log of the specific
    viscosity.
    """
    return torch.log(eta_sp / eta_sp_range.min) / np.log(
        eta_sp_range.max / eta_sp_range.min
    )


def preprocess_eta_sp(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Add noise, cap the values, take the log, then normalize."""
    eta_sp += (
        eta_sp * 0.05 * torch.normal(torch.zeros_like(eta_sp), torch.ones_like(eta_sp))
    )
    eta_sp = torch.fmin(eta_sp, torch.tensor(eta_sp_range.max))
    eta_sp = torch.fmax(eta_sp, torch.tensor(eta_sp_range.min))
    # eta_sp[eta_sp == eta_sp_range.max] = eta_sp_range.min
    return normalize_eta_sp(eta_sp, eta_sp_range)
