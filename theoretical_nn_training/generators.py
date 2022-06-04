from typing import Iterable, Tuple

import numpy as np
import torch

from theoretical_nn_training.datatypes import Param, Resolution

PHI = Param(1e-6, 1e-3)
NW = Param(100, 3e5)
ETA_SP = Param(1, 4e6)

BG = Param(0.3, 1.1)
BTH = Param(0.2, 0.7)
PE = Param(10, 18)

SHIFT = 1e-4


def get_Bth(Bg: torch.Tensor) -> torch.Tensor:
    Bth = 0.54 * Bg + 0.05
    Bth += Bth * 0.05 * torch.normal(torch.zeros_like(Bth), torch.ones_like(Bth))
    return Bth


def normalize_params(
    Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """Simple linear normalization."""
    Bg = (Bg - BG.min - SHIFT) / (BG.max - BG.min)
    Bth = (Bth - BTH.min - SHIFT) / (BTH.max - BTH.min)
    Pe = (Pe - PE.min - SHIFT) / (PE.max - PE.min)
    return Bg, Bth, Pe


def unnormalize_params(
    Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor
) -> Tuple[torch.Tensor, ...]:
    """Simple linear normalization."""
    Bg = Bg * (BG.max - BG.min) + BG.min + SHIFT
    Bth = Bth * (BTH.max - BTH.min) + BTH.min + SHIFT
    Pe = Pe * (PE.max - PE.min) + PE.min + SHIFT
    return Bg, Bth, Pe


def preprocess_visc(eta_sp: torch.Tensor) -> torch.Tensor:
    """Add noise, cap the values, take the log, then normalize."""
    eta_sp += (
        eta_sp * 0.05 * torch.normal(torch.zeros_like(eta_sp), torch.ones_like(eta_sp))
    )
    eta_sp = torch.fmin(eta_sp, torch.tensor(ETA_SP.max))
    eta_sp = torch.fmax(eta_sp, torch.tensor(ETA_SP.min))
    return normalize_visc(eta_sp)


def normalize_visc(eta_sp: torch.Tensor) -> torch.Tensor:
    return torch.log(eta_sp / ETA_SP.min) / np.log(ETA_SP.max / ETA_SP.min)


def surface_generator(
    num_batches: int, batch_size: int, device: torch.device, resolution: Resolution
) -> Iterable[Tuple[torch.Tensor, ...]]:
    """Generate `batch_size` surfaces, based on ranges for `Bg`, `Bth`, and
    `Pe`, to be used in a `for` loop.

    It defines the resolution of the surface based on either user input
    (keyword argument `resolution`). It then generates random values for `Bg`,
    `Bth`, and `Pe`, evaluates the `(phi, Nw, eta_sp)` surface, and normalizes
    the result. The normalized values of `eta_sp` and `(Bg, Bth, Pe)` are
    yielded as `X` and `y` for use in a neural network.

    Input:
        `num_batches` (`int`) : The number of loops to be iterated through.
        `batch_size` (`int`) : The length of the generated values.
        `device` (`torch.device`): The device to do computations on.
        `resolution` (tuple of `int`s) : The shape of the last two dimensions
            of the generated values.

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Generated,
            normalized values of `eta_sp` at indexed `phi` and `Nw`.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """

    # Create tensors for phi (concentration) and Nw (chain length)
    # Both are meshed and tiled to cover a 3D tensor of size
    # (batch_size, *resolution) for simple, element-wise operations
    phi = torch.tensor(
        np.geomspace(PHI.min, PHI.max, resolution.phi, endpoint=True),
        dtype=torch.float,
        device=device,
    )

    Nw = torch.tensor(
        np.geomspace(NW.min, NW.max, resolution.Nw, endpoint=True),
        dtype=torch.float,
        device=device,
    )

    phi, Nw = torch.meshgrid(phi, Nw, indexing="xy")
    phi = torch.tile(phi, (batch_size, 1, 1))
    Nw = torch.tile(Nw, (batch_size, 1, 1))

    def generate_surfaces(
        Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor
    ) -> torch.Tensor:
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth**6 / phi**2)

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling
        Ne = (
            Pe**2
            * g
            * torch.fmin(
                torch.tensor([1], device=device),
                torch.fmin(
                    (Bth / Bg) ** (2 / (6 * 0.588 - 3)) / Bth**2,
                    Bth**4 * phi ** (2 / 3),
                ),
            )
        )

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover at c = c**
        eta_sp = Nw * (1 + (Nw / Ne) ** 2) * torch.fmin(1 / g, phi / Bth**2)

        return eta_sp

    for _ in range(num_batches):
        y = torch.rand((batch_size, 3), device=device, dtype=torch.float)
        Bg, Bth, Pe = unnormalize_params(*(y.T))
        eta_sp = generate_surfaces(Bg, Bth, Pe)
        X = preprocess_visc(eta_sp).to(torch.float)
        yield X, y


def voxel_image_generator(
    num_batches: int, batch_size: int, device: torch.device, resolution: Resolution
) -> Iterable[Tuple[torch.Tensor, ...]]:
    """Uses surface_generator to generate a surface with a resolution one more,
    then generates a 3D binary array dictating whether or not the surface
    passes through a given voxel. This is determined using the facts that:
     - The surface is continuous
     - The surface monotonically increases with increasing phi and Nw
     - phi and Nw increase with increasing index
    If the voxel corner at index (i, j, k+1) is greater than the surface value
    at (i, j), and if the corner at index (i+1, j+1, k) is less than the value
    at (i+1, j+1), then the surface passes through.
    Input:
        `num_batches` (`int`) : The number of loops to be iterated through.
        `batch_size` (`int`) : The length of the generated values.
        `device` (`torch.device`): The device to do computations on.
        `resolution` (tuple of `int`s) : The shape of the last three dimensions
            of the generated values.

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Binary array
            dictating whether or not the surface passes through the indexed
            voxel.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """
    s_res = Resolution(resolution.phi + 1, resolution.Nw + 1)

    eta_sp = preprocess_visc(
        torch.tensor(
            np.geomspace(ETA_SP.min, ETA_SP.max, resolution.eta_sp + 1, endpoint=True),
            dtype=torch.float,
            device=device,
        )
    )

    for X, y in surface_generator(num_batches, batch_size, device, resolution=s_res):
        surf = torch.tile(
            X.reshape((batch_size, *s_res, 1)), (1, 1, 1, resolution.eta_sp + 1)
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surf[:, :-1, :-1, :-1] < eta_sp[1:], surf[:, 1:, 1:, 1:] > eta_sp[:-1]
        ).to(dtype=torch.float, device=device)

        yield image, y
