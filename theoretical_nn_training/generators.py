r"""This file contains two efficient generator clasees based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from a choice of `Uniform`,
`LogNormal`, or `Beta` distributions.

TODO: update documentation for mode and strip_nw
"""
from typing import Protocol, Tuple

import numpy as np
import scipy.interpolate
import torch
from typing_extensions import Self

import theoretical_nn_training.data_processing as data
from theoretical_nn_training.data_processing import Mode, Range, Resolution


class Config(Protocol):
    """The protocol for a configuration class used by the generators. For a full
    example, see configuration.NNConfig.
    """

    device: torch.device
    resolution: Resolution
    mode: Mode
    phi_range: Range
    nw_range: Range
    eta_sp_range: Range
    bg_range: Range
    bth_range: Range
    pe_range: Range
    num_nw_strips: int
    batch_size: int


class Generator(Protocol):
    """Protocol for two generators: the SurfaceGenerator, which yields 2D
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
    """Returns a callable, iterable object that can be used in a `for` loop to
    efficiently generate polymer solution specific viscosity data as a function of
    concentration and weight-average degree of polymerization as defined by three
    parameters: the blob parameters $B_g$ and $B_{th}$, and the entanglement packing
    number $P_e$.

    Usage:
    ```
        config = data_processing.NNConfig('configurations/sample_config.yaml')
        generator = SurfaceGenerator(config)
        for surfaces, features in generator(num_batches):
            ...
    ```

    Input:
        `config` (`data_processing.NNConfig`) : The configuration object.
            Specifically, the generator uses the attributes specified in the Config
            protocol in this module.
        `strip_nw` (`bool`, default `False`) : If true, then before the generated
            surfaces are returned, they are stripped down in the Nw dimension to reflect
            that experiments are usually done with fewer than 8 different molecular
            weight species. All specific viscosity data is set to 0 for all values of Nw
            except for several randomly chosen values. Then, the remaining values are
            interpolated to cover the whole intermediate range.
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
        ).reshape((1, config.resolution.phi, 1))

        self.Nw = torch.tensor(
            np.geomspace(
                config.nw_range.min,
                config.nw_range.max,
                config.resolution.Nw,
                endpoint=True,
            ),
            dtype=torch.float,
            device=config.device,
        ).reshape((1, 1, config.resolution.Nw))

        if config.mode is not Mode.THETA:
            self.bg_distribution = data.feature_distribution(
                config.bg_range, config.device
            )
        if config.mode is not Mode.GOOD:
            self.bth_distribution = data.feature_distribution(
                config.bth_range, config.device
            )
        self.pe_distribution = data.feature_distribution(config.pe_range, config.device)

        if config.mode is Mode.GOOD:
            self.generation_function = self._good_generation
        elif config.mode is Mode.MIXED:
            self.generation_function = self._mixed_generation
        else:
            self.generation_function = self._theta_generation

        # self.xi is used in the 2D interpolation after stripping the surface of most
        # Nw values. It is here so it will only be computed once.
        if config.num_nw_strips:
            phi_mesh, Nw_mesh = torch.meshgrid(
                self.phi.ravel(), self.Nw.ravel(), indexing="xy"
            )
            self.flattened_phi_Nw_mesh = torch.repeat_interleave(
                torch.stack((phi_mesh.ravel(), Nw_mesh.ravel()), dim=1),
                config.batch_size,
                dim=0,
            )
            print(self.flattened_phi_Nw_mesh.size())
            del phi_mesh, Nw_mesh

        self.config = config

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

        surfaces, features = self.generation_function()

        if not self.config.num_nw_strips:
            return surfaces, features

        # Stripping the Nw dimension to simulate experimental data, which usually
        # only has a handful of different samples. Note: this `randint` call is a random
        # sample with replacement, meaning the total number of strips may be less than
        # `self.config.num_nw_strips`
        nw_index_choices = torch.randint(
            0,
            self.config.resolution.Nw,
            torch.Size((self.config.batch_size, 1, self.config.num_nw_strips)),
            device=self.config.device,
        )
        for surface, nw_index_choice in zip(surfaces, nw_index_choices):
            new_surface = torch.zeros_like(surface)
            new_surface[:, nw_index_choice] = surface[:, nw_index_choice]
            surface = new_surface

        # Interpolate stripped data
        ravelled_surfaces = surfaces.ravel()
        have_values = torch.logical_and(
            (0 < ravelled_surfaces), (ravelled_surfaces < 1)
        )

        interp_surfaces = scipy.interpolate.griddata(
            points=self.flattened_phi_Nw_mesh[have_values],
            values=ravelled_surfaces[have_values],
            xi=self.flattened_phi_Nw_mesh,
            method="linear",
            fill_value=0.0,
        )

        return (
            torch.tensor(interp_surfaces, dtype=torch.float, device=self.config.device),
            features,
        )

    def _good_generation(self) -> Tuple[torch.Tensor, torch.Tensor]:

        Bg = self.bg_distribution.sample(torch.Size((self.config.batch_size, 1, 1)))
        Pe = self.pe_distribution.sample(torch.Size((self.config.batch_size, 1, 1)))

        # Number of repeat units per correlation blob
        g = Bg ** (3 / 0.764) / self.phi ** (1 / 0.764)

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover where correlation length equals Kuhn length.
        # Ne is simply Pe^2 * g
        eta_sp = (
            self.Nw
            * (1 + (self.Nw / Pe**2 / g) ** 2)
            * torch.fmin(
                1 / g,
                self.phi / Bg ** (1 / (1 - 0.588)),
            )
        )

        surfaces = data.preprocess_eta_sp(eta_sp, self.config.eta_sp_range)
        features = torch.stack(
            (
                data.normalize_feature(Bg.ravel(), self.config.bg_range),
                data.normalize_feature(Pe.ravel(), self.config.pe_range),
            ),
            dim=1,
        ).to(self.config.device)

        return surfaces, features

    def _mixed_generation(self) -> Tuple[torch.Tensor, torch.Tensor]:

        Bg = self.bg_distribution.sample(torch.Size((self.config.batch_size, 1, 1)))
        Pe = self.pe_distribution.sample(torch.Size((self.config.batch_size, 1, 1)))

        # To ensure that this model doesn't generate the athermal condition
        # (see _good_generatrion), we select Bth uniformly between 0 and Bg^(1/0.824)
        # We also ensure that Bth stays within the Range.
        Bth = (
            torch.rand(self.config.batch_size, 1, 1, device=self.config.device)
            * (self.config.bth_range.max - self.config.bth_range.min)
            + self.config.bth_range.min
        ) * Bg ** (1 / 0.824)

        # Number of repeat units per correlation blob
        # The minimum function accounts for the piecewise crossover at the thermal
        # blob overlap concentration.
        g = torch.fmin(
            Bg ** (3 / 0.764) / self.phi ** (1 / 0.764),
            Bth**6 / self.phi**2,
        )

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling.
        # Corresponding concentration ranges are listed next to the expressions.
        Ne = Pe**2 * torch.fmin(
            torch.fmin(
                Bth ** (0.944 / 0.528) / Bg ** (2 / 0.528) * g,  # c* < c < c_th
                Bth**2 * self.phi ** (-4 / 3),  # c_th < c < b^-3
            ),
            g,  # b^-3 < c
        )

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover where correlation length equals Kuhn length.
        eta_sp = (
            self.Nw * (1 + (self.Nw / Ne) ** 2) * torch.fmin(1 / g, self.phi / Bth**2)
        )

        surfaces = data.preprocess_eta_sp(eta_sp, self.config.eta_sp_range)
        features = torch.stack(
            (
                data.normalize_feature(Bg.ravel(), self.config.bg_range),
                data.normalize_feature(Bth.ravel(), self.config.bth_range),
                data.normalize_feature(Pe.ravel(), self.config.pe_range),
            ),
            dim=1,
        ).to(self.config.device)

        return surfaces, features

    def _theta_generation(self) -> Tuple[torch.Tensor, torch.Tensor]:

        Bth = self.bth_distribution.sample(torch.Size((self.config.batch_size, 1, 1)))
        Pe = self.pe_distribution.sample(torch.Size((self.config.batch_size, 1, 1)))

        # Number of repeat units per correlation blob
        g = Bth**6 / self.phi**2

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling.
        # Corresponding concentration ranges are listed next to the expressions.
        Ne = Pe**2 * torch.fmin(
            Bth**2 * self.phi ** (-4 / 3),  # c* < c < b^-3
            g,  # b^-3 < c
        )

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover where correlation length equals Kuhn length.
        eta_sp = (
            self.Nw * (1 + (self.Nw / Ne) ** 2) * torch.fmin(1 / g, self.phi / Bth**2)
        )

        surfaces = data.preprocess_eta_sp(eta_sp, self.config.eta_sp_range)
        features = torch.stack(
            (
                data.normalize_feature(Bth.ravel(), self.config.bth_range),
                data.normalize_feature(Pe.ravel(), self.config.pe_range),
            ),
            dim=1,
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
            Specifically, the generator uses the attributes specified in the Config
            protocol in this module.
        `strip_nw` (`bool`, default `False`) : If true, then before the generated
            surfaces are returned, they are stripped down in the Nw dimension to reflect
            that experiments are usually done with fewer than 8 different molecular
            weight species. All specific viscosity data is set to 0 for all values of Nw
            except for several randomly chosen values. Then, the remaining values are
            interpolated to cover the whole intermediate range.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.resolution = Resolution(
            config.resolution.phi + 1,
            config.resolution.Nw + 1,
            config.resolution.eta_sp + 1,
        )

        self.grid_values = data.preprocess_eta_sp(
            torch.tensor(
                np.geomspace(
                    config.eta_sp_range.min,
                    config.eta_sp_range.max,
                    config.resolution.eta_sp,
                    endpoint=True,
                ),
                dtype=torch.float,
                device=config.device,
            ),
            config.eta_sp_range,
        )

        self.surface_generator = SurfaceGenerator(self.config)

    def __call__(self, num_batches: int) -> Self:
        self.num_batches = num_batches
        return self

    def __iter__(self) -> Self:
        self._surface_generator_iter = iter(self.surface_generator(self.num_batches))
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        surfaces, features = next(self._surface_generator_iter)

        surfaces = torch.tile(
            surfaces.reshape(
                (
                    self.config.batch_size,
                    self.config.resolution.phi,
                    self.config.resolution.Nw,
                    1,
                )
            ),
            (1, 1, 1, self.config.resolution.eta_sp),
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surfaces[:, :-1, :-1, :-1] < self.grid_values[1:],
            surfaces[:, 1:, 1:, 1:] > self.grid_values[:-1],
        ).to(dtype=torch.int16, device=self.config.device)
        # TODO: check that this works with int16

        return image, features
