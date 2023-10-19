r"""Efficiently generate batched samples of molecular parameters in a uniform
distribution, construct specific viscosity as a function of concentration and 
degree of polymerization of the polymer solution from the samples of molecular
parameters, and yield the normalized values (between 0 and 1).
"""
from __future__ import annotations
import logging
from math import log10
from typing import Optional

import torch

import psst


def normalize(
    arr: torch.Tensor, min: float, max: float, log_scale: bool = False
) -> torch.Tensor:
    r"""Normalize a Tensor from its true values into a :math:`[0, 1]` scale using
    given minimum and maximum values following the equation for each value of
    :math:`x` in ``arr``:

    .. math::

        y = \frac{x - min}{max - min}

    If ``log_scale`` is ``True``, the equation is instead

    .. math::

        y = \frac{\log_{10}(x) - \log_{10}(min)}{\log_{10}(max) - \log_{10}(min)}

    Args:
        arr (torch.Tensor): The Tensor to be normalized.
        min (float): The value to map to 0.
        max (float): The value to map to 1.
        log_scale (bool, optional): When ``True``, the base-10 logarithm of the values
          of ``arr``, ``min``, and ``max`` are used instead of the given values.
          Defaults to ``False``.

    Returns:
        torch.Tensor: The normalized Tensor.
    """
    out_arr = arr.clone()
    if log_scale:
        min = log10(min)
        max = log10(max)
        out_arr.log10_()

    out_arr -= min
    out_arr /= max - min
    return out_arr


def unnormalize(
    arr: torch.Tensor, min: float, max: float, log_scale: bool = False
) -> torch.Tensor:
    r"""Unnormalize a Tensor from the :math:`[0, 1]` scale to its true values using
    given minimum and maximum values following the equation for each value of
    :math:`x` in ``arr``:

    .. math::

        y = (max - min) x + min

    If ``log_scale`` is ``True``, the equation is instead

    .. math::

        \begin{align*}
        y^\prime &= \left[\log_{10}(max) - \log_{10}(min)\right] x + \log_{10}(min) \\
        y &= 10^{y^\prime}
        \end{align*}

    Args:
        arr (torch.Tensor): The Tensor to be unnormalized.
        min (float): The value to map to from 0.
        max (float): The value to map to from 1.
        log_scale (bool, optional): When ``True``, the base-10 logarithm of the values
          of ``arr``, ``min``, and ``max`` are used instead of the given values.
          Defaults to ``False``.

    Returns:
        torch.Tensor: The unnormalized Tensor.
    """
    out_arr = arr.clone()
    if log_scale:
        min = log10(min)
        max = log10(max)

    out_arr = arr * (max - min)
    out_arr += min

    if log_scale:
        torch.pow(10, out_arr, out=out_arr)
    return out_arr


class SampleGenerator:
    """Procedurally generates batches of viscosity curves.

    The resulting object is callable and iterable with similar functionality to the
    built-in ``range`` function. It takes one parameter, the number of batches/cycles,
    and the four element tuple it generates consists of 

    1. The normalized, reduced viscosity with shape
    ``(batch_size, phi_range.num, nw_range.num)``. This is considered as a batch of
    2D images that can be used to train a neural network (e.g.,
    ``psst.models.Inception3``).

    2. The generated values of :math:`B_g` with shape ``(batch_size,)``.
    
    3. The generated values of :math:`B_{th}` (same shape as for :math:`B_g`).
    
    4. The generated values of :math:`P_e` (same shape again).

    Example:
        >>> import psst
        >>> from psst.models import Inception3
        >>> 
        >>> model = Inception3()
        >>> config = psst.getConfig("config.yaml")
        >>> gen_samples = psst.SampleGenerator(**config.generator_config)
        >>> num_batches = (
        ...     config.run_config.num_samples_test
        ...     // config.generator_config.batch_size
        ... )
        >>> for viscosity, bg, bth, pe in gen_samples(num_batches):
        >>>     pred_bg = model(viscosity)

    Args:
        batch_size (int): The number of values of Bg, Bth, and Pe (and thus
          the number of viscosity curves) to generate.
        parameter (:class:`Parameter`): Either ``"Bg"`` or ``"Bth"`` for good
          solvent behavior or thermal blob behavior, respectively.
        phi_range (:class:`Range`): The min, max and number of reduced
          concentration values to use for the viscosity curves.
        nw_range (:class:`Range`): As with ``phi_range``, but for values of degree
          of polymerization.
        visc_range (:class:`Range`): The minimum and maximum values of viscosity
          to use for normalization.
        bg_range (:class:`Range`): The minimum and maximum values of the good
          solvent blob parameter to use for normalization and generation.
        bth_range (:class:`Range`): The minimum and maximum values of the thermal
          blob parameter to use for normalization and generation.
        pe_range (:class:`Range`): The minimum and maximum values of the
          entanglement packing number to use for normalization and generation.
        device (torch.device, optional): Device on which to create batches and compute
          samples. Defaults to ``torch.device("cpu")``.
        generator (torch.Generator, optional): Random number generator to use for
          values of :math:`Bg`, :math:`Bth`, and :math:`Pe`. Most useful during
          testing, allowing a fixed seed to be used. A value of ``None`` creates a
          generic torch.Generator instance. Defaults to ``None``.
    """

    def __init__(
        self,
        *,
        parameter: psst.Parameter,
        phi_range: psst.Range,
        nw_range: psst.Range,
        visc_range: psst.Range,
        bg_range: psst.Range,
        bth_range: psst.Range,
        pe_range: psst.Range,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self._log = logging.getLogger("psst.main")
        self._log.info("Initializing SampleGenerator")
        self._log.debug("SampleGenerator: device = %s", str(device))

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

        assert isinstance(parameter, psst.Parameter)
        if parameter == "Bg":
            # self.primary_B = self.Bg
            self._other_B = self.bth
            self._get_single_samples = self._get_bg_samples
            self.denominator = self.nw * self.phi ** (1 / 0.764)
            self._log.debug("Initialized Bg-specific members")
        else:
            # self.primary_B = self.Bth
            self._other_B = self.bg
            self._get_single_samples = self._get_bth_samples
            self.denominator = self.nw * self.phi**2
            self._log.debug("Initialized Bth-specific members")

        # Create tensors for phi (concentration) and Nw (number of repeat units per
        # chain). Both are broadcastable to size
        # (batch_size, phi_range.num, nw_range.num) for element-wise operations
        if phi_range.log_scale:
            self.phi = torch.logspace(
                log10(phi_range.min),
                log10(phi_range.max),
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
                log10(nw_range.min),
                log10(nw_range.max),
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

        self.norm_visc_range = psst.Range(
            visc_range.min / self.denominator.max(),
            visc_range.max / self.denominator.min(),
        )

        self._num_batches: int = 0
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
        self._num_batches = num_batches
        return self

    def __iter__(self):
        self._index = 0
        self._log.info("Starting %d iterations", self._num_batches)
        return self

    def __next__(
        self,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._index >= self._num_batches:
            self._log.info("Completed all batches")
            raise StopIteration

        self._index += 1
        self._log.debug("Generating batch %6d/%d", self._index, self._num_batches)

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

        self.visc[is_combo] = self._get_combo_samples(
            self.bg[is_combo], self.bth[is_combo], self.pe[is_combo]
        )
        self._log.debug("Computed combo samples")
        self.visc[~is_combo] = self._get_single_samples(
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

    def _get_combo_samples(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
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

    def _get_bg_samples(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = (Bg**3 / self.phi) ** (1 / 0.764)
        Ne = Pe**2 * g
        return self.nw / g * (1 + (self.nw / Ne)) ** 2

    def _get_bth_samples(self, Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor):
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
        samples: torch.Tensor,
        max_num_rows_select: int = 12,
        max_num_rows_nonzero: int = 48,
        num_concentrations_per_sample: int = 65,
    ) -> torch.Tensor:
        num_batches = samples.shape[0]
        indices = torch.arange(self.nw.size(2), device=self.device)

        self._log.debug("Trimming Nw rows")
        for i in range(num_batches):
            num_nonzero_per_row = torch.sum(samples[i] > 0, dim=0)
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
            samples[i, deselected_rows, :] = 0.0

        self._log.debug("Trimming phi rows")
        for i in range(num_batches):
            nonzero_rows = samples[i].nonzero(as_tuple=True)[0]
            deselected_rows = torch.randint(
                nonzero_rows.min(),
                nonzero_rows.max(),
                size=(num_concentrations_per_sample,),
                device=self.device,
                generator=self.generator,
            )
            samples[i, deselected_rows, :] = 0.0

        return samples
