r"""This file contains two efficient generator clasees based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from a choice of `Uniform`,
`LogNormal`, or `Beta` distributions.
"""
from typing import Protocol, Tuple

import numpy as np
import torch
from typing_extensions import Self

import theoretical_nn_training.data_processing as data

# TODO: Add options for generators for slices of Nw


class Config(Protocol):
    device: torch.device
    resolution: data.Resolution
    phi_range: data.Range
    nw_range: data.Range
    eta_sp_range: data.Range
    bg_range: data.Range
    bth_range: data.Range
    pe_range: data.Range
    batch_size: int


class Generator(Protocol):
    """Abstract base class for two generators: the SurfaceGenerator, which yields 2D
    representations of surfaces, and the VoxelImageGenerator, which yields a 3D image of
    the surfaces produced by SurfaceGenerator. These generators are meant to be run on
    high-performance devices, such as a CUDA-enabled GPU.
    """

    def __init__(self, config: Config) -> None:
        ...

    def __call__(self, num_batches: int) -> Self:
        ...

    def __iter__(self) -> Self:
        ...

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class SurfaceGenerator:
    """Returns a callable object that can be used in a `for` loop to efficiently
    generate polymer solution specific viscosity data as a function of concentration
    and weight-average degree of polymerization as defined by three parameters: the
    blob parameters $B_g$ and $B_{th}$, and the entanglement packing number $P_e$.

    Usage:
    ```
    config = data_processing.NNConfig('configurations/sample_config.yaml')
    generator = SurfaceGenerator(config)
    for surfaces, features in generator(num_batches):
        ...
    ```

    Input:
        `config` (`data_processing.NNConfig`) : The configuration object.
            Specifically, the generator uses the attributes device, resolution,
            phi_range, nw_range, eta_sp_range, bg_range, bth_range, pe_range, and
            batch_size.
    """

    def __init__(self, config: Config) -> None:

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
            device=config.device,
        )

        self.Nw = torch.tensor(
            np.geomspace(
                config.nw_range.min,
                config.nw_range.max,
                config.resolution.Nw,
                endpoint=True,
            ),
            dtype=torch.float,
            device=config.device,
        )

        self.phi_mesh, self.Nw_mesh = torch.meshgrid(self.phi, self.Nw, indexing="xy")
        self.phi_mesh = torch.tile(self.phi_mesh, (config.batch_size, 1, 1))
        self.Nw_mesh = torch.tile(self.Nw_mesh, (config.batch_size, 1, 1))
        self.config = config

        self.bg_distribution = data.feature_distribution(config.bg_range)
        self.bth_distribution = data.feature_distribution(config.bth_range)
        self.pe_distribution = data.feature_distribution(config.pe_range)

    def __call__(self, num_batches: int) -> Self:
        self.num_batches = num_batches
        return self

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

        Bg, Bth, Pe = data.unnormalize_features(
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
            self.config.device
        )
        Bth = torch.tile(Bth.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.config.device
        )
        Pe = torch.tile(Pe.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.config.device
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

        surfaces = data.preprocess_eta_sp(eta_sp, self.config.eta_sp_range)
        features = torch.stack(
            (normalized_Bg, normalized_Bth, normalized_Pe), dim=1
        ).to(self.config.device)

        return surfaces, features


class VoxelImageGenerator:
    """Returns a callable object that can be used in a `for` loop to efficiently
    generate 3D images of polymer solution specific viscosity data as a function of
    concentration and weight-average degree of polymerization as defined by three
    parameters: the blob parameters $B_g$ and $B_{th}$, and the entanglement packing
    number $P_e$. Internally, this iteratres over an instance of SurfaceGenerator.

    Usage:
    ```
    config = data_processing.NNConfig('configurations/sample_config.yaml')
    generator = VoxelImageGenerator(config)
    for surfaces, features in generator(num_batches):
        ...
    ```

    Input:
        `config` (`data_processing.NNConfig`) : The configuration object.
            Specifically, the generator uses the attributes device, resolution,
            phi_range, nw_range, eta_sp_range, bg_range, bth_range, pe_range, and
            batch_size.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.resolution = data.Resolution(
            config.resolution.phi + 1,
            config.resolution.Nw + 1,
            config.resolution.eta_sp,
        )

        self.eta_sp = data.preprocess_eta_sp(
            torch.tensor(
                np.geomspace(
                    config.eta_sp_range.min,
                    config.eta_sp_range.max,
                    config.resolution.eta_sp + 1,
                    endpoint=True,
                ),
                dtype=torch.float,
                device=config.device,
            ),
            config.eta_sp_range,
        )

        self.config = config

    def __call__(self, num_batches: int) -> Self:
        self.num_batches = num_batches
        return self

    def __iter__(self) -> Self:
        self._surface_generator = iter(SurfaceGenerator(self.config)(self.num_batches))

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
        ).to(dtype=torch.float, device=self.config.device)

        return image, features
