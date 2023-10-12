from typing import Iterable, Protocol

import torch

from ..temp_st.core import data_processing as data
from ..temp_st.core import configuration as config
from ..temp_st.core.surface_generator import SurfaceGenerator


class Generator(Protocol):
    """Protocol for two generators: the SurfaceGenerator, which yields 2D
    representations of surfaces, and the VoxelImageGenerator, which yields a 3D image of
    the surfaces produced by SurfaceGenerator. These generators are meant to be run on
    high-performance devices, such as a CUDA-enabled GPU.
    """

    def __init__(self, config: config.Configuration) -> None:
        ...

    def __call__(self, num_batches: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        ...

    def __iter__(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        ...

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...


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
    """

    def __init__(
        self, config: config.Configuration, device: torch.device, mode: str = "combo"
    ) -> None:
        self.config = config

        self.grid_values = config.eta_sp_range.get(device=device)
        self.grid_values = data.preprocess_eta_sp(self.grid_values, config.eta_sp_range)

        self.surface_generator = SurfaceGenerator(self.config)

    def __call__(self, num_batches: int):
        self.num_batches = num_batches
        return self

    def __iter__(self):
        self._surface_generator_iter = iter(self.surface_generator(self.num_batches))
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
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
        ).to(dtype=torch.float, device=self.config.device)

        return image, features
