r"""This file contains two efficient generator clasees based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from 3 uniform distributions.
"""
import logging

import numpy as np
import torch

from psst.configuration import GeneratorConfig, ParameterChoice


def normalize(arr: torch.Tensor, min: float, max: float, log_scale: bool = False):
    out_arr = arr.clone()
    if log_scale:
        min = np.log10(min)
        max = np.log10(max)
        out_arr.log10_()
    
    out_arr -= min
    out_arr /= (max - min)
    return out_arr

def unnormalize(arr: torch.Tensor, min: float, max: float, log_scale: bool = False):
    out_arr = arr.clone()
    if log_scale:
        min = np.log10(min)
        max = np.log10(max)
        torch.pow(10, out_arr, out=out_arr)
    
    out_arr = arr * (max - min)
    out_arr += min
    return out_arr

class SurfaceGenerator:
    """Returns a callable, iterable object that can be used in a `for` loop to
    efficiently generate polymer solution specific viscosity data as a function of
    concentration and weight-average degree of polymerization as defined by three
    parameters: the blob parameters $B_g$ and $B_{th}$, and the entanglement packing
    number $P_e$.

    Most tensors in this class are broadcastable to size
    `(batch_size, phi_range.num, nw_range.num)`
    for simple, element-wise operations.

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
    """

    def __init__(
        self, config: GeneratorConfig, device: torch.device
    ) -> None:
        self.config = config
        self.device = device
        self.parameter = config.parameter

        self.log = logging.getLogger("psst.main")
        self.log.info("Initializing SurfaceGenerator")
        self.log.debug("SurfaceGenerator: device = %s; config = %s", str(self.device), str(config))
        
        self.batch_size = config.batch_size
        self.num_batches: int = 0
        self._index: int = 0
        self.rng = torch.Generator(device=self.device)
        self.log.debug("Initialized random number generator")
        
        # Create tensors for phi (concentration) and Nw (number of repeat units per chain)
        # Both are broadcastable to size (batch_size, phi_range.shape[0], nw_range.shape[0])
        # for simple, element-wise operations
        self.phi = config.phi_range.to(device=device).reshape((1, -1, 1))
        self.Nw = config.nw_range.to(device=device).reshape((1, 1, -1))

        self.log.debug("Initialized self.phi with size %s", str(self.phi.shape))
        self.log.debug("Initialized self.Nw with size %s", str(self.Nw.shape))

        self.Bg = torch.zeros((self.batch_size, 1, 1), dtype=torch.float32, device=device)
        self.Bth = torch.zeros_like(self.Bg)
        self.Pe = torch.zeros_like(self.Bg)

        self.log.debug("Initialized self.Bg, self.Bth, self.Pe each with size %s", str(self.Bg.shape))

        assert isinstance(config.parameter, ParameterChoice)
        if config.parameter == ParameterChoice.Bg:
            # self.primary_B = self.Bg
            self.other_B = self.Bth
            self._get_single_surfaces = self._get_bg_surfaces
            self.denom = self.Nw * self.phi**(1/0.764)
            self.log.debug("Initialized Bg-specific members")
        else:
            # self.primary_B = self.Bth
            self.other_B = self.Bg
            self._get_single_surfaces = self._get_bth_surfaces
            self.denom = self.Nw * self.phi**2
            self.log.debug("Initialized Bth-specific members")

        self.visc_min = config.eta_sp_dist.min / self.denom.max()
        self.visc_max = config.eta_sp_dist.max / self.denom.min()
        self.visc = torch.zeros(
            (self.batch_size, self.phi.shape[1], self.Nw.shape[2]),
            dtype=torch.float32,
            device=device,
        )

        self.log.debug("Completed initialization")

    def __call__(self, num_batches: int):
        self.num_batches = num_batches
        return self

    def __iter__(self):
        self._index = 0
        self.log.info("Starting %d iterations", self.num_batches)
        return self

    def __next__(self) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._index >= self.num_batches:
            self.log.info("Completed all batches")
            raise StopIteration
    
        self._index += 1
        self.log.debug("Generating batch %6d/%d", self._index, self.num_batches)

        if self.visc.ndim == 4:
            self.visc.squeeze_()

        self.Bg.uniform_(self.config.bg_dist.min, self.config.bg_dist.max)
        self.Bth.uniform_(self.config.bth_dist.min, self.config.bth_dist.max)
        self.Pe.uniform_(self.config.pe_dist.min, self.config.pe_dist.max)

        self.log.debug("Sampled values for Bg, Bth, Pe")

        is_combo = torch.randint(
            2, size=(self.batch_size,), device=self.device, generator=self.rng, dtype=torch.bool
        )
        self.other_B.data[~is_combo] = 0.0

        self.log.debug("Chose combo and single samples")

        self.visc[is_combo] = self._get_combo_surfaces(
            self.Bg[is_combo],
            self.Bth[is_combo],
            self.Pe[is_combo]
        )
        self.log.debug("Computed combo samples")
        self.visc[~is_combo] = self._get_single_surfaces(
            self.Bg[~is_combo],
            self.Bth[~is_combo],
            self.Pe[~is_combo]
        )
        self.log.debug("Computed single samples")

        self.visc = self._trim(self.visc) / self.denom
        self.log.debug("Trimmed and divided samples")

        normalize(self.visc, self.visc_min, self.visc_max, log_scale=True)
        normalize(self.Bg, self.config.bg_dist.min, self.config.bg_dist.max)
        normalize(self.Bth, self.config.bth_dist.min, self.config.bth_dist.max)
        normalize(self.Pe, self.config.pe_dist.min, self.config.pe_dist.max)
        self.log.debug("Normalized results")

        return self.visc, self.Bg.flatten(), self.Bth.flatten(), self.Pe.flatten()

    def _get_combo_surfaces(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = torch.minimum(
            (Bg**3 / self.phi) **(1/0.764),
            Bth**6 / self.phi**2
        )
        Ne = Pe**2 * torch.minimum(
            Bg**(0.056 / (0.528*0.764)) * Bth**(0.944/0.528) / self.phi ** (1/0.764),
            torch.minimum(
                (Bth / self.phi**(2/3))**2,
                (Bth**3 / self.phi)**2
            )
        )
        return self.Nw * (1 + (self.Nw / Ne))**2 * torch.minimum(
            1/g,
            self.phi / Bth**2
        )
    
    def _get_bg_surfaces(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = (Bg**3 / self.phi) ** (1/0.764)
        Ne = Pe**2 * g
        return self.Nw / g * (1 + (self.Nw / Ne))**2
    
    def _get_bth_surfaces(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = Bth**6 / self.phi**2
        Ne = Pe**2 * torch.minimum(
            (Bth / self.phi**(2/3))**2,
            (Bth**3 / self.phi)**2
        )
        return self.Nw * (1 + (self.Nw / Ne))**2 * torch.minimum(
            1/g,
            self.phi / Bth**2
        )

    # @torch.compile
    def _trim(
        self,
        surfaces: torch.Tensor,
        max_num_rows_select: int = 12,
        max_num_rows_nonzero: int = 48,
        num_concentrations_per_surface: int = 65,
    ) -> torch.Tensor:

        num_batches = surfaces.shape[0]
        indices = torch.arange(self.Nw.size(2), device=self.device)

        self.log.debug("Trimming Nw rows")
        for i in range(num_batches):
            num_nonzero_per_row = torch.sum(surfaces[i] > 0, dim=0)
            top_rows = torch.argsort(num_nonzero_per_row, descending=True, dim=0)[:max_num_rows_nonzero]
            selected = torch.randint(0, top_rows.shape[0], 
                size=(max_num_rows_select,),
                device=self.device,
                generator=self.rng,
            )
            deselected_rows = torch.isin(indices, top_rows[selected], invert=True)
            surfaces[i, deselected_rows, :] = 0.0

        # num_nonzero_per_row = torch.sum(surfaces > 0, dim=1)
        # top_rows = torch.argsort(num_nonzero_per_row, descending=True, dim=1)[
        #     :, :max_num_rows_nonzero
        # ]
        # selected = torch.randint(
        #     0,
        #     max_num_rows_nonzero,
        #     size=(self.batch_size, max_num_rows_select),
        #     device=self.device,
        #     generator=self.rng,
        # )
        # selected_rows = 
        # for b, sel in enumerate(selected):
        #     surfaces[b, sel, :] = 0.0

        # deselected_rows = torch.tensor(
        #     [
        #         torch.isin(indices, rows[sel], invert=True)
        #         for rows, sel in zip(top_rows, selected)
        #     ],
        #     dtype=torch.bool,
        #     device=self.device,
        # )

        # surfaces[deselected_rows] = 0.0

        self.log.debug("Trimming phi rows")
        for i in range(num_batches):
            nonzero_rows = surfaces[i].nonzero(as_tuple=True)[0]
            deselected_rows = torch.randint(
                nonzero_rows.min(),
                nonzero_rows.max(),
                size=(num_concentrations_per_surface,),
                device=self.device,
                generator=self.rng,
            )
            surfaces[i, deselected_rows, :] = 0.0

        return surfaces
