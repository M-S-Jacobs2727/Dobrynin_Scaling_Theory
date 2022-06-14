from abc import ABC
from typing import Iterable, Tuple

import numpy as np
import torch
from typing_extensions import Self

import theoretical_nn_training.data_processing as data
from theoretical_nn_training.configuration import NNConfig

# TODO: Add options for generators or new generators for non-Pe training


class Generator(ABC):
    """Abstract base class for two generators: the SurfaceGenerator, which yields 2D
    representations of surfaces, and the VoxelImageGenerator, which yields a 3D image of
    the surfaces produced by SurfaceGenerator. These generators are meant to be run on
    high-performance devices, such as a CUDA-enabled GPU.
    """

    def __call__(self, num_batches: int):
        self.num_batches = num_batches
        return self

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class SurfaceGenerator(Generator):
    def __init__(self, device: torch.device, config: NNConfig) -> None:
        """Returns a callable object that can be used in a `for` loop to efficiently
        generate polymer solution specific viscosity data as a function of concentration
        and weight-average degree of polymerization as defined by three parameters: the
        blob parameters $B_g$ and $B_{th}$, and the entanglement packing number $P_e$.

        Usage:
        ```
        config = data_processing.NNConfig('configurations/sample_config.yaml')
        generator = SurfaceGenerator(torch.device('cuda'), config)
        for surfaces, features in generator(num_batches):
            ...
        ```

        Input:
            `device` (`torch.device`) : The device on which the computations proceed.
                Preferrably a GPU.
            `config` (`data_processing.NNConfig`) : The configuration object.
                Specifically, the generator uses the attributes resolution, phi_range,
                nw_range, eta_sp_range, bg_range, bth_range, pe_range, and batch_size.
        """

        # Create tensors for phi (concentration) and Nw (chain length)
        # Both are meshed and tiled to cover a 3D tensor of size
        # (batch_size, *resolution) for simple, element-wise operations
        self.phi = torch.tensor(
            np.geomspace(
                config.phi_range.min,
                config.phi_range.max,
                config.resolution.phi,
                endpoint=True,
            ),
            dtype=torch.float,
            device=device,
        )

        self.Nw = torch.tensor(
            np.geomspace(
                config.nw_range.min,
                config.nw_range.max,
                config.resolution.Nw,
                endpoint=True,
            ),
            dtype=torch.float,
            device=device,
        )

        self.phi_mesh, self.Nw_mesh = torch.meshgrid(self.phi, self.Nw, indexing="xy")
        self.phi_mesh = torch.tile(self.phi_mesh, (config.batch_size, 1, 1))
        self.Nw_mesh = torch.tile(self.Nw_mesh, (config.batch_size, 1, 1))
        self.config = config
        self.device = device

        self.bg_distribution = data.param_dist(config.bg_range)
        self.bth_distribution = data.param_dist(config.bth_range)
        self.pe_distribution = data.param_dist(config.pe_range)

    def __iter__(self) -> Self:
        self._index = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._index >= self.num_batches:
            raise StopIteration
        self._index += 1

        normalized_Bg = self.bg_distribution.sample(
            torch.Size((self.config.batch_size,))
        )
        normalized_Bth = self.bth_distribution.sample(
            torch.Size((self.config.batch_size,))
        )
        normalized_Pe = self.pe_distribution.sample(
            torch.Size((self.config.batch_size,))
        )

        Bg, Bth, Pe = data.unnormalize_params(
            normalized_Bg,
            normalized_Bth,
            normalized_Pe,
            self.config.bg_range,
            self.config.bth_range,
            self.config.pe_range,
        )

        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        # TODO: check if there is a way to do operations without the meshes. It may be
        # more readable this way, but it is less memory-efficient.
        shape = torch.Size((1, *(self.phi_mesh.size()[1:])))
        Bg = torch.tile(Bg.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.device
        )
        Bth = torch.tile(Bth.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.device
        )
        Pe = torch.tile(Pe.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.device
        )

        # Number of repeat units per correlation blob
        # The minimum function accounts for the piecewise crossover at the thermal
        # blob overlap concentration.
        g = torch.fmin(
            Bg ** (3 / 0.764) / self.phi_mesh ** (1 / 0.764),
            Bth**6 / self.phi_mesh**2,
        )

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling.
        # Corresponding concentration ranges are listed next to the expressions.
        Ne = Pe**2 * torch.fmin(
            torch.fmin(
                g * (Bth / Bg) ** (2 / (6 * 0.588 - 3)) / Bth**4,  # c < c_th
                Bth**2 * self.phi_mesh ** (-4 / 3),  # c_th < c < b^-3
            ),
            g,  # b^-3 < c
        )

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover where correlation length equals Kuhn length.
        eta_sp = (
            self.Nw_mesh
            * (1 + (self.Nw_mesh / Ne) ** 2)
            * torch.fmin(1 / g, self.phi_mesh / Bth**2)
        )

        surfaces = data.preprocess_visc(eta_sp, self.config.eta_sp_range)
        features = torch.stack(
            (normalized_Bg, normalized_Bth, normalized_Pe), dim=1
        ).to(self.device)

        return surfaces, features


class VoxelImageGenerator(Generator):
    def __init__(self, device: torch.device, config: NNConfig) -> None:
        """Returns a callable object that can be used in a `for` loop to efficiently
        generate 3D images of polymer solution specific viscosity data as a function of
        concentration and weight-average degree of polymerization as defined by three
        parameters: the blob parameters $B_g$ and $B_{th}$, and the entanglement packing
        number $P_e$. Internally, this iteratres over an instance of SurfaceGenerator.

        Usage:
        ```
        config = data_processing.NNConfig('configurations/sample_config.yaml')
        generator = VoxelImageGenerator(torch.device('cuda'), config)
        for surfaces, features in generator(num_batches):
            ...
        ```

        Input:
            `device` (`torch.device`) : The device on which the computations proceed.
                Preferrably a GPU.
            `config` (`data_processing.NNConfig`) : The configuration object.
                Specifically, the generator uses the attributes resolution, phi_range,
                nw_range, eta_sp_range, bg_range, bth_range, pe_range, and batch_size.
        """
        self.config = config
        self.config.resolution = data.Resolution(
            config.resolution.phi + 1,
            config.resolution.Nw + 1,
            config.resolution.eta_sp,
        )

        self.eta_sp = data.preprocess_visc(
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

        self._surface_generator = SurfaceGenerator(device, config)
        self.device = device
        self.config = config

    def __iter__(self) -> Self:
        self._surface_generator = iter(
            SurfaceGenerator(self.device, self.config)(self.num_batches)
        )

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            surfaces, features = next(self._surface_generator)
        except StopIteration:
            raise StopIteration

        surfaces = torch.tile(
            surfaces.reshape(
                (
                    self.config.batch_size,
                    self.config.resolution.phi,
                    self.config.resolution.Nw,
                    1,
                )
            ),
            (1, 1, 1, self.config.resolution.eta_sp + 1),
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surfaces[:, :-1, :-1, :-1] < self.eta_sp[1:],
            surfaces[:, 1:, 1:, 1:] > self.eta_sp[:-1],
        ).to(dtype=torch.float, device=self.device)

        return image, features


def surface_generator(
    num_batches: int,
    device: torch.device,
    config: NNConfig,
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
        # TODO: probability distributions!!!
        normalized_Bg, normalized_Bth, normalized_Pe = torch.rand(
            (3, config.batch_size), device=device, dtype=torch.float
        )
        Bg, Bth, Pe = data.unnormalize_params(
            normalized_Bg,
            normalized_Bth,
            normalized_Pe,
            config.bg_range,
            config.bth_range,
            config.pe_range,
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

        surfaces = data.preprocess_visc(eta_sp, config.eta_sp_range).to(torch.float)
        features = torch.stack((normalized_Bg, normalized_Bth, normalized_Pe), dim=1)

        yield surfaces, features


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
    surface_resolution = data.Resolution(
        config.resolution.phi + 1, config.resolution.Nw + 1
    )

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
        surface = torch.tile(
            X.reshape(
                (config.batch_size, surface_resolution.phi, surface_resolution.Nw, 1)
            ),
            (1, 1, 1, config.resolution.eta_sp + 1),
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surface[:, :-1, :-1, :-1] < eta_sp[1:], surface[:, 1:, 1:, 1:] > eta_sp[:-1]
        ).to(dtype=torch.float, device=device)

        yield image, y
