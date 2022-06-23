r"""This file contains two efficient generator clasees based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from a choice of `Uniform`,
`LogNormal`, or `Beta` distributions.

TODO: update documentation for mode and strip_nw
"""
from typing import Protocol, Tuple

import numpy as np
import torch
from typing_extensions import Self

import theoretical_nn_training.data_processing as data
from theoretical_nn_training.data_processing import FeatureRange, Mode, SurfaceRange


class Config(Protocol):
    """The protocol for a configuration class used by the generators. For a full
    example, see configuration.NNConfig.
    """

    device: torch.device
    mode: Mode
    phi_range: SurfaceRange
    nw_range: SurfaceRange
    eta_sp_range: SurfaceRange
    bg_range: FeatureRange
    bth_range: FeatureRange
    pe_range: FeatureRange
    batch_size: int


class Generator(Protocol):
    """Protocol for two generators: the SurfaceGenerator, which yields 2D
    representations of surfaces, and the VoxelImageGenerator, which yields a 3D image of
    the surfaces produced by SurfaceGenerator. These generators are meant to be run on
    high-performance devices, such as a CUDA-enabled GPU.
    """

    def __init__(self, config: Config, strip_nw: bool) -> None:
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

    def __init__(self, config: Config, strip_nw: bool = False) -> None:

        # Create tensors for phi (concentration) and Nw (chain length)
        # Both are meshed and tiled to cover a 3D tensor of size
        # (batch_size, *resolution) for simple, element-wise operations
        self.phi = torch.tensor(
            np.geomspace(
                config.phi_range.min,
                config.phi_range.max,
                config.phi_range.resolution,
                endpoint=True,
            ),
            dtype=torch.float,
            device=config.device,
        )

        self.Nw = torch.tensor(
            np.geomspace(
                config.nw_range.min,
                config.nw_range.max,
                config.nw_range.resolution,
                endpoint=True,
            ),
            dtype=torch.float,
            device=config.device,
        )

        self.phi_mesh, self.Nw_mesh = torch.meshgrid(self.phi, self.Nw, indexing="xy")
        self.phi_mesh = torch.tile(self.phi_mesh, (config.batch_size, 1, 1))
        self.Nw_mesh = torch.tile(self.Nw_mesh, (config.batch_size, 1, 1))

        self.bg_distribution = data.feature_distribution(
            config.bg_range, config.batch_size
        )
        self.bth_distribution = data.feature_distribution(
            config.bth_range, config.batch_size
        )
        self.pe_distribution = data.feature_distribution(
            config.pe_range, config.batch_size
        )

        self.config = config
        self.strip_nw = strip_nw

        if self.config.mode is Mode.GOOD:
            self.generation_function = self._good_generation
        elif self.config.mode is Mode.MIXED:
            self.generation_function = self._mixed_generation
        else:
            self.generation_function = self._theta_generation

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

        if not self.strip_nw:
            return surfaces, features

        # Stripping the Nw dimension to simulate experimental data, which usually
        # only has a handful of different samples.
        # TODO: add interpolation of the few samples, as we will for experimental data
        nw_index_choices = torch.randint(
            0,
            self.config.nw_range.resolution,
            torch.Size((self.config.batch_size, 8)),
            device=self.config.device,
        )
        to_keep = torch.zeros(
            (self.config.batch_size, self.config.nw_range.resolution), dtype=torch.bool
        )
        for surface, to_keep_b, choices_b in zip(surfaces, to_keep, nw_index_choices):
            to_keep_b[choices_b] = True
            surface[~to_keep_b] = 0

        return surfaces, features

    def _good_generation(self) -> Tuple[torch.Tensor, torch.Tensor]:

        normalized_Bg = self.bg_distribution.sample()
        normalized_Pe = self.pe_distribution.sample()

        Bg = data.unnormalize_feature(normalized_Bg, self.config.bg_range)
        Pe = data.unnormalize_feature(normalized_Pe, self.config.pe_range)

        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        # TODO: check if there is a way to do operations without the meshes. It may be
        # more readable this way, but it is less memory-efficient.
        shape = torch.Size((1, *(self.phi_mesh.size()[1:])))
        Bg = torch.tile(Bg.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.config.device
        )
        Pe = torch.tile(Pe.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.config.device
        )

        # Number of repeat units per correlation blob
        g = Bg ** (3 / 0.764) / self.phi_mesh ** (1 / 0.764)

        # Number of repeat units per entanglement strand
        # Kavassalis-Noolandi only
        Ne = Pe**2 * g

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover where correlation length equals Kuhn length.
        eta_sp = (
            self.Nw_mesh
            * (1 + (self.Nw_mesh / Ne) ** 2)
            * torch.fmin(1 / g, self.phi_mesh / Bg ** (1 / (1 - 0.588)))
        )

        surfaces = data.preprocess_eta_sp(eta_sp, self.config.eta_sp_range)
        features = torch.stack((normalized_Bg, normalized_Pe), dim=1).to(
            self.config.device
        )

        return surfaces, features

    def _mixed_generation(self) -> Tuple[torch.Tensor, torch.Tensor]:

        normalized_Bg = self.bg_distribution.sample()
        # normalized_Bth = self.bth_distribution.sample()
        normalized_Pe = self.pe_distribution.sample()

        Bg = data.unnormalize_feature(normalized_Bg, self.config.bg_range)
        # Bth = data.unnormalize_feature(normalized_Bth, self.config.bth_range)
        Pe = data.unnormalize_feature(normalized_Pe, self.config.pe_range)

        # To ensure that this model doesn't generate the athermal condition
        # (see _good_generatrion), we select Bth uniformly between 0 and Bg^(1/0.824)
        # We also ensure that Bth stays within the Range.
        Bth = (
            torch.rand(torch.Size((self.config.batch_size,)))
            * Bg ** (1 / 0.824)
            * (self.config.bth_range.max - self.config.bth_range.min)
            + self.config.bth_range.min
        )
        normalized_Bth = data.normalize_feature(Bth, self.config.bth_range)

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
                Bth ** (0.944 / 0.528) / Bg ** (2 / 0.528) * g,  # c* < c < c_th
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

    def _theta_generation(self) -> Tuple[torch.Tensor, torch.Tensor]:

        normalized_Bth = self.bth_distribution.sample()
        normalized_Pe = self.pe_distribution.sample()

        Bth = data.unnormalize_feature(normalized_Bth, self.config.bth_range)
        Pe = data.unnormalize_feature(normalized_Pe, self.config.pe_range)

        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        # TODO: check if there is a way to do operations without the meshes. It may be
        # more readable this way, but it is less memory-efficient.
        shape = torch.Size((1, *(self.phi_mesh.size()[1:])))
        Bth = torch.tile(Bth.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.config.device
        )
        Pe = torch.tile(Pe.reshape((self.config.batch_size, 1, 1)), shape).to(
            self.config.device
        )

        # Number of repeat units per correlation blob
        g = Bth**6 / self.phi_mesh**2

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling.
        # Corresponding concentration ranges are listed next to the expressions.
        Ne = Pe**2 * torch.fmin(
            Bth**2 * self.phi_mesh ** (-4 / 3),  # c* < c < b^-3
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
        features = torch.stack((normalized_Bth, normalized_Pe), dim=1).to(
            self.config.device
        )

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

    def __init__(self, config: Config, strip_nw: bool = False) -> None:
        self.config = config
        self.config.nw_range.resolution += 1
        self.config.phi_range.resolution += 1
        self.config.eta_sp_range.resolution += 1

        self.grid_values = data.preprocess_eta_sp(
            torch.tensor(
                np.geomspace(
                    config.eta_sp_range.min,
                    config.eta_sp_range.max,
                    config.eta_sp_range.resolution,
                    endpoint=True,
                ),
                dtype=torch.float,
                device=config.device,
            ),
            config.eta_sp_range,
        )

        self.strip_nw = strip_nw
        self.surface_generator = SurfaceGenerator(self.config, self.strip_nw)

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
                    self.config.phi_range.resolution,
                    self.config.nw_range.resolution,
                    1,
                )
            ),
            (1, 1, 1, self.config.eta_sp_range.resolution),
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surfaces[:, :-1, :-1, :-1] < self.grid_values[1:],
            surfaces[:, 1:, 1:, 1:] > self.grid_values[:-1],
        ).to(dtype=torch.int16, device=self.config.device)
        # TODO: check that this works with int16

        return image, features
