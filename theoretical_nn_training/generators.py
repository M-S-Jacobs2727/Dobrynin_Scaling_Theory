from typing import Iterable, Tuple

import numpy as np
import torch

import theoretical_nn_training.data_processing as data
from theoretical_nn_training.data_processing import Resolution


def surface_generator(
    num_batches: int, config: data.Config
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
        np.geomspace(
            config.phi_param.min,
            config.phi_param.max,
            config.resolution.phi,
            endpoint=True,
        ),
        dtype=torch.float,
        device=config.device,
    )

    Nw = torch.tensor(
        np.geomspace(
            config.nw_param.min,
            config.nw_param.max,
            config.resolution.Nw,
            endpoint=True,
        ),
        dtype=torch.float,
        device=config.device,
    )

    phi, Nw = torch.meshgrid(phi, Nw, indexing="xy")
    phi = torch.tile(phi, (config.batch_size, 1, 1))
    Nw = torch.tile(Nw, (config.batch_size, 1, 1))

    def generate_surfaces(
        Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor
    ) -> torch.Tensor:
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((config.batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((config.batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((config.batch_size, 1, 1)), shape)

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
                torch.tensor([1], device=config.device),
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
        y0, y1, y2 = torch.rand(
            (3, config.batch_size), device=config.device, dtype=torch.float
        )
        Bg, Bth, Pe = data.unnormalize_params(
            y0, y1, y2, config.bg_param, config.bth_param, config.pe_param
        )
        eta_sp = generate_surfaces(Bg, Bth, Pe)
        X = data.preprocess_visc(eta_sp, config.eta_sp_param).to(torch.float)
        y = torch.cat((y0, y1, y2), 1)
        yield X, y


def voxel_image_generator(
    num_batches: int, config: data.Config
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
    s_res = Resolution(config.resolution.phi + 1, config.resolution.Nw + 1)

    eta_sp = data.preprocess_visc(
        torch.tensor(
            np.geomspace(
                config.eta_sp_param.min,
                config.eta_sp_param.max,
                config.resolution.eta_sp + 1,
                endpoint=True,
            ),
            dtype=torch.float,
            device=config.device,
        ),
        config.eta_sp_param,
    )

    for X, y in surface_generator(num_batches, config):
        surf = torch.tile(
            X.reshape((config.batch_size, *s_res, 1)),
            (1, 1, 1, config.resolution.eta_sp + 1),
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surf[:, :-1, :-1, :-1] < eta_sp[1:], surf[:, 1:, 1:, 1:] > eta_sp[:-1]
        ).to(dtype=torch.float, device=config.device)

        yield image, y
