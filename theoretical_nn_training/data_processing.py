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

import torch
import torch.distributions


class Mode(Enum):
    """An enum for selecting the features to use and train against.

    Values:
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
    feature_range: Range, device: torch.device
) -> torch.distributions.Distribution:
    """Returns a generator that can be sampled in batches of size `batch_size` for the
    features Bg, Bth, or Pe. These can be the Beta distribution (if feature_range.alpha
    and feature_range.beta are defined), the LogNormal distribution (if feature_range.mu
    and feature_range.sigma are defined), or the Uniform distribution otherwise.

    Note: to use these in a neural network, they must first be normalized using, e.g.,
    `data_processing.normalize_feature`.
    """
    if feature_range.alpha and feature_range.beta:
        return torch.distributions.Beta(
            torch.tensor(feature_range.alpha, dtype=torch.float, device=device),
            torch.tensor(feature_range.beta, dtype=torch.float, device=device),
        )
    if feature_range.mu and feature_range.sigma:
        return torch.distributions.LogNormal(
            torch.tensor(feature_range.mu, dtype=torch.float, device=device),
            torch.tensor(feature_range.sigma, dtype=torch.float, device=device),
        )
    return torch.distributions.Uniform(
        torch.tensor(feature_range.min, dtype=torch.float, device=device),
        torch.tensor(feature_range.max, dtype=torch.float, device=device),
    )


def get_Bth_from_Bg(Bg: torch.Tensor) -> torch.Tensor:
    """This computes Bth from Bg on the assumption that they have a linear
    relationship.
    """
    Bth = 0.54 * Bg + 0.05
    Bth += Bth * 0.1 * torch.normal(torch.zeros_like(Bth), torch.ones_like(Bth))
    return Bth


def normalize_feature(feature, feature_range: Range):
    """Performs simple linear normalization."""
    return (feature - feature_range.min) / (feature_range.max - feature_range.min)


def unnormalize_feature(feature, feature_range: Range):
    """Inverts simple linear normalization."""
    return feature * (feature_range.max - feature_range.min) + feature_range.min


def unnormalize_eta_sp(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Inverts a simple linear normalization of the specific viscosity."""
    return (
        torch.exp(
            eta_sp
            * torch.log(
                torch.tensor(eta_sp_range.max / eta_sp_range.min, device=eta_sp.device)
            )
        )
        * eta_sp_range.min
    )


def normalize_eta_sp(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Performs a simple linear normalization of the natural log of the specific
    viscosity.
    """
    return torch.log(eta_sp / eta_sp_range.min) / torch.log(
        torch.tensor(eta_sp_range.max / eta_sp_range.min, device=eta_sp.device)
    )


def preprocess_eta_sp(eta_sp: torch.Tensor, eta_sp_range: Range) -> torch.Tensor:
    """Add noise, cap the values, take the log, then normalize."""
    eta_sp += (
        eta_sp * 0.05 * torch.normal(torch.zeros_like(eta_sp), torch.ones_like(eta_sp))
    )
    eta_sp = torch.fmin(eta_sp, torch.tensor(eta_sp_range.max))
    eta_sp = torch.fmax(eta_sp, torch.tensor(eta_sp_range.min))
    eta_sp[eta_sp == eta_sp_range.max] = eta_sp_range.min
    return normalize_eta_sp(eta_sp, eta_sp_range)
