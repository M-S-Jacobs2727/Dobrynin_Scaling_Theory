r"""This file contains two efficient generator clasees based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from 3 uniform distributions.
"""
import logging

import numpy as np
import torch

from psst.configuration import Parameter, Range


def normalize(arr: torch.Tensor, min: float, max: float, log_scale: bool = False):
    out_arr = arr.clone()
    if log_scale:
        min = np.log10(min)
        max = np.log10(max)
        out_arr.log10_()

    out_arr -= min
    out_arr /= max - min
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
    """Creates a callable, iterable object to procedurally generate viscosity
    curves as functions of concentration (`phi`) and chain degree of polymerization
    (`nw`).

    Usage:

    >>> generator = SurfaceGenerator("Bg", batch_size=64, device=torch.device("cuda"))
    >>> for viscosity, bg, bth, pe in generator(512000):
    >>>     pred_bg = model(viscosity)
    >>>     loss = loss_fn(bg, pred_bg)

    :param parameter: Either "Bg" or "Bth" for good solvent behavior for thermal
        behavior, respectively
    :type parameter: :class:`psst.configuration.Parameter`
    :param phi_range: The min, max and number of reduced concentration values to use
        for the viscosity curves
    :type phi_range: :class:`psst.configuration.Range`
    :param nw_range: As with `phi_range`, but for values of degree of polymerization
    :type nw_range: :class:`psst.configuration.Range`
    :param visc_range: The minimum and maximum values of viscosity to use for
    normalization
    :type visc_range: :class:`psst.configuration.Range`
    :param bg_range: The minimum and maximum values of the good solvent blob
        parameter to use for normalization and generation.
    :type bg_range: :class:`psst.configuration.Range`
    :param bth_range: The minimum and maximum values of the thermal blob
        parameter to use for normalization and generation.
    :type bth_range: :class:`psst.configuration.Range`
    :param pe_range: The minimum and maximum values of the entanglement packing
        number to use for normalization and generation.
    :type pe_range: :class:`psst.configuration.Range`
    :param batch_size: The number of values of Bg, Bth, and Pe (and thus the number
        of viscosity curves) to generate, defaults to 1.
    :type batch_size: int, optional
    :param device: _description_, defaults to torch.device("cpu")
    :type device: `torch.device`, optional
    :param generator: _description_, defaults to None
    :type generator: `torch.Generator`, optional
    """

    def __init__(
        self,
        parameter: Parameter,
        *,
        phi_range: Range,
        nw_range: Range,
        visc_range: Range,
        bg_range: Range,
        bth_range: Range,
        pe_range: Range,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        generator: torch.Generator | None = None,
    ) -> None:
        
        self._log = logging.getLogger("psst.main")
        self._log.info("Initializing SurfaceGenerator")
        self._log.debug("SurfaceGenerator: device = %s", str(device))

        self.parameter = parameter
        self.batch_size = batch_size

        self.phi_range = phi_range
        self.nw_range = nw_range
        self.visc_range = visc_range
        self.bg_range = bg_range
        self.bth_range = bth_range
        self.pe_range = pe_range

        self.device = device
        if generator is None:
            self.generator = torch.Generator(device=self.device)
        else:
            self.generator = generator
        self._log.debug("Initialized random number generator")

        assert isinstance(parameter, Parameter)
        if parameter == "Bg":
            # self.primary_B = self.Bg
            self._other_B = self.bth
            self._get_single_surfaces = self._get_bg_surfaces
            self.denominator = self.nw * self.phi ** (1 / 0.764)
            self._log.debug("Initialized Bg-specific members")
        else:
            # self.primary_B = self.Bth
            self._other_B = self.bg
            self._get_single_surfaces = self._get_bth_surfaces
            self.denominator = self.nw * self.phi**2
            self._log.debug("Initialized Bth-specific members")

        # Create tensors for phi (concentration) and Nw (number of repeat units per chain)
        # Both are broadcastable to size (batch_size, phi_range.shape[0], nw_range.shape[0])
        # for simple, element-wise operations
        if phi_range.log_scale:
            self.phi = torch.logspace(
                np.log10(phi_range.min),
                np.log10(phi_range.max),
                phi_range.num,
                device=device,
            )
        else:
            self.phi = torch.linspace(
                phi_range.min, phi_range.max, phi_range.num, device=device
            )
        self.phi = self.phi.reshape(1, -1, 1)

        if nw_range.log_scale:
            self.nw = torch.logspace(
                np.log10(nw_range.min),
                np.log10(nw_range.max),
                nw_range.num,
                device=device,
            )
        else:
            self.nw = torch.linspace(
                nw_range.min, nw_range.max, nw_range.num, device=device
            )
        self.nw = self.nw.reshape(1, 1, -1)
        self._log.debug("Initialized self.phi with size %s", str(self.phi.shape))
        self._log.debug("Initialized self.nw with size %s", str(self.nw.shape))

        self.visc = torch.zeros(
            (self.batch_size, self.phi.shape[1], self.nw.shape[2]),
            dtype=torch.float32,
            device=device,
        )

        self.norm_visc_range = Range(
            visc_range.min / self.denominator.max(),
            visc_range.max / self.denominator.min(),
        )

        self.num_batches: int = 0
        self._index: int = 0

        self.bg = torch.zeros(
            (self.batch_size, 1, 1), dtype=torch.float32, device=device
        )
        self.bth = torch.zeros_like(self.bg)
        self.pe = torch.zeros_like(self.bg)
        self._log.debug(
            "Initialized self.bg, self.bth, self.pe each with size %s",
            str(self.bg.shape),
        )

        self._log.debug("Completed initialization")

    def __call__(self, num_batches: int):
        self.num_batches = num_batches
        return self

    def __iter__(self):
        self._index = 0
        self._log.info("Starting %d iterations", self.num_batches)
        return self

    def __next__(
        self,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._index >= self.num_batches:
            self._log.info("Completed all batches")
            raise StopIteration

        self._index += 1
        self._log.debug("Generating batch %6d/%d", self._index, self.num_batches)

        if self.visc.ndim == 4:
            self.visc.squeeze_()

        self.bg.uniform_(self.bg_range.min, self.bg_range.max)
        self.bth.uniform_(self.bth_range.min, self.bth_range.max)
        self.pe.uniform_(self.pe_range.min, self.pe_range.max)

        self._log.debug("Sampled values for Bg, Bth, Pe")

        is_combo = torch.randint(
            2,
            size=(self.batch_size,),
            device=self.device,
            generator=self.generator,
            dtype=torch.bool,
        )
        self._other_B.data[~is_combo] = 0.0

        self._log.debug("Chose combo and single samples")

        self.visc[is_combo] = self._get_combo_surfaces(
            self.bg[is_combo], self.bth[is_combo], self.pe[is_combo]
        )
        self._log.debug("Computed combo samples")
        self.visc[~is_combo] = self._get_single_surfaces(
            self.bg[~is_combo], self.bth[~is_combo], self.pe[~is_combo]
        )
        self._log.debug("Computed single samples")

        self.visc = self._trim(self.visc) / self.denominator
        self._log.debug("Trimmed and divided samples")

        normalize(
            self.visc,
            self.norm_visc_range.min,
            self.norm_visc_range.max,
            log_scale=True,
        )
        normalize(self.bg, self.bg_range.min, self.bg_range.max)
        normalize(self.bth, self.bth_range.min, self.bth_range.max)
        normalize(self.pe, self.pe_range.min, self.pe_range.max)
        self._log.debug("Normalized results")

        return self.visc, self.bg.flatten(), self.bth.flatten(), self.pe.flatten()

    def _get_combo_surfaces(
        self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor
    ):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = torch.minimum((Bg**3 / self.phi) ** (1 / 0.764), Bth**6 / self.phi**2)
        Ne = Pe**2 * torch.minimum(
            Bg ** (0.056 / (0.528 * 0.764))
            * Bth ** (0.944 / 0.528)
            / self.phi ** (1 / 0.764),
            torch.minimum((Bth / self.phi ** (2 / 3)) ** 2, (Bth**3 / self.phi) ** 2),
        )
        return (
            self.nw
            * (1 + (self.nw / Ne)) ** 2
            * torch.minimum(1 / g, self.phi / Bth**2)
        )

    def _get_bg_surfaces(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = (Bg**3 / self.phi) ** (1 / 0.764)
        Ne = Pe**2 * g
        return self.nw / g * (1 + (self.nw / Ne)) ** 2

    def _get_bth_surfaces(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = Bth**6 / self.phi**2
        Ne = Pe**2 * torch.minimum(
            (Bth / self.phi ** (2 / 3)) ** 2, (Bth**3 / self.phi) ** 2
        )
        return (
            self.nw
            * (1 + (self.nw / Ne)) ** 2
            * torch.minimum(1 / g, self.phi / Bth**2)
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
        indices = torch.arange(self.nw.size(2), device=self.device)

        self._log.debug("Trimming Nw rows")
        for i in range(num_batches):
            num_nonzero_per_row = torch.sum(surfaces[i] > 0, dim=0)
            top_rows = torch.argsort(num_nonzero_per_row, descending=True, dim=0)[
                :max_num_rows_nonzero
            ]
            selected = torch.randint(
                0,
                top_rows.shape[0],
                size=(max_num_rows_select,),
                device=self.device,
                generator=self.generator,
            )
            deselected_rows = torch.isin(indices, top_rows[selected], invert=True)
            surfaces[i, deselected_rows, :] = 0.0

        self._log.debug("Trimming phi rows")
        for i in range(num_batches):
            nonzero_rows = surfaces[i].nonzero(as_tuple=True)[0]
            deselected_rows = torch.randint(
                nonzero_rows.min(),
                nonzero_rows.max(),
                size=(num_concentrations_per_surface,),
                device=self.device,
                generator=self.generator,
            )
            surfaces[i, deselected_rows, :] = 0.0

        return surfaces
