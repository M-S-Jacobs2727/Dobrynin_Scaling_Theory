from typing import Iterable, Tuple

import numpy as np
import torch

import theoretical_nn_training.data_processing as data
from theoretical_nn_training.configuration import NNConfig

# TODO: Add options for generators or new generators for non-Pe training


def surface_generator(
    num_batches: int, device: torch.device, config: NNConfig
) -> Iterable[Tuple[torch.Tensor, ...]]:
    """Generate `batch_size` surfaces, based on ranges for `Bg`, `Bth`, and
    `Pe`, to be used in a `for` loop.

    It generates random values for `Bg`, `Bth`, and `Pe`, evaluates the
    `(phi, Nw, eta_sp)` surface, and normalizes the result. The normalized values of
    `eta_sp` and `(Bg, Bth, Pe)` are yielded as `X` and `y` for use in a neural network.

    Input:
        `num_batches` (`int`) : The number of loops to be iterated through.
        `device` (`torch.device`): The device to do computations on.
        `config` (`data_processing.NNConfig`) :

    Output:
        `X` (`torch.Tensor` of size `(batch_size, resolution.phi, resolution.Nw)`) :
            Generated, normalized values of `eta_sp` at indexed `phi` and `Nw` defined
            by `Param` values in `config`.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """

    # Create tensors for phi (concentration) and Nw (chain length)
    # Both are meshed and tiled to cover a 3D tensor of size
    # (batch_size, *resolution) for simple, element-wise operations
    # TODO GQ: hi I have a problem: while you need to call those variables their greek
    # letter representation for the config, we're no longer there; you also
    # literally name the variables above what they do. Highly recommended to change
    # references to greek stuff to literally what they are, since it's more human
    # readable that way.
    phi = torch.tensor(
        np.geomspace(
            config.phi_range.min,
            config.phi_range.max,
            config.resolution.phi,
            endpoint=True,
        ),
        dtype=torch.float,
        device=device,
    )

    Nw = torch.tensor(
        np.geomspace(
            config.nw_range.min,
            config.nw_range.max,
            config.resolution.Nw,
            endpoint=True,
        ),
        dtype=torch.float,
        device=device,
    )

    phi, Nw = torch.meshgrid(phi, Nw, indexing="xy")
    phi = torch.tile(phi, (config.batch_size, 1, 1))
    Nw = torch.tile(Nw, (config.batch_size, 1, 1))

    # TODO GQ: So same thing here, with the params here; is it possible to use a human
    # understandable word/words for each of these?

    for _ in range(num_batches):
        # TODO GQ: Moar readability pl0x, for the entire for loop; wtf is an X or a y
        y0, y1, y2 = torch.rand(
            (3, config.batch_size), device=device, dtype=torch.float
        )
        Bg, Bth, Pe = data.unnormalize_params(
            y0, y1, y2, config.bg_range, config.bth_range, config.pe_range
        )

        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((config.batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((config.batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((config.batch_size, 1, 1)), shape)

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # TODO GQ: Above line: What does c < c** actually mean in english? Maybe worth
        # translating it, then adding (c < c**) to the end of it, to demonstrate
        # the part of the formula this is dealing w/
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
        X = data.preprocess_visc(eta_sp, config.eta_sp_range).to(torch.float)
        y = torch.stack((y0, y1, y2), dim=1)
        yield X, y


def voxel_image_generator(
    num_batches: int, device: torch.device, config: NNConfig
) -> Iterable[Tuple[torch.Tensor, ...]]:
    """Uses `surface_generator` to generate a surface with a resolution one greater,
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
        `device` (`torch.device`): The device to do computations on.
        `config` (`data_processing.NNConfig`) :

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Binary array
            dictating whether or not the surface passes through the indexed
            voxel.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """
    s_res = data.Resolution(config.resolution.phi + 1, config.resolution.Nw + 1)

    eta_sp = data.preprocess_visc(
        torch.tensor(
            np.geomspace(
                config.eta_sp_range.min,
                config.eta_sp_range.max,
                config.resolution.eta_sp + 1,
                endpoint=True,
            ),
            dtype=torch.float,
            device=device,
        ),
        config.eta_sp_range,
    )

    for X, y in surface_generator(num_batches, device, config):
        surf = torch.tile(
            X.reshape((config.batch_size, s_res.phi, s_res.Nw, 1)),
            (1, 1, 1, config.resolution.eta_sp + 1),
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surf[:, :-1, :-1, :-1] < eta_sp[1:], surf[:, 1:, 1:, 1:] > eta_sp[:-1]
        ).to(dtype=torch.float, device=device)

        yield image, y
