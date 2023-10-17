r"""Efficiently generate batched samples of molecular parameters in a uniform
distribution, construct specific viscosity as a function of concentration and 
degree of polymerization of the polymer solution from the samples of molecular
parameters, and yield the normalized values (between 0 and 1).
"""
import logging
from math import log10

import torch

import psst


def normalize(
    arr: torch.Tensor, min: float, max: float, log_scale: bool = False
) -> torch.Tensor:
    r"""Normalize a Tensor from its true values into a [0, 1] scale using given minimum
    and maximum values following the equation for each value of :math:`x` in `arr`:

    ..math::
        y = (x - min) / (max - min)

    If `log_scale` is `True`, the equation is instead

    ..math::
        y = (\log_{10}(x) - \log_{10}(min)) / (\log_{10}(max) - \log_{10}(min))

    :param arr: The Tensor to be normalized
    :type arr: torch.Tensor
    :param min: The value to map to 0
    :type min: float
    :param max: The value to map to 1
    :type max: float
    :param log_scale: When True, the base-10 logarithm of the values of `arr`, `min`,
    and `max` are used instead of the given values, defaults to False
    :type log_scale: bool, optional
    :return: The normalized Tensor
    :rtype: torch.Tensor
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
    r"""Unnormalize a Tensor from the [0, 1] scale to its true values using given minimum
    and maximum values following the equation for each value of :math:`x` in `arr`:

    ..math::
        y = x (max - min) + min

    If `log_scale` is `True`, the equation is instead

    ..math::
        y^\prime = x (\log_{10}(max) - \log_{10}(min)) + \log_{10}(min)
        y = 10^{y^\prime}

    :param arr: The Tensor to be unnormalized
    :type arr: torch.Tensor
    :param min: The value to map to from 0
    :type min: float
    :param max: The value to map to from 1
    :type max: float
    :param log_scale: When True, the base-10 logarithm of the values of `arr`, `min`,
    and `max` are used instead of the given values, defaults to False
    :type log_scale: bool, optional
    :return: The unnormalized Tensor
    :rtype: torch.Tensor
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
    """Creates a callable, iterable object to procedurally generate viscosity
    curves as functions of concentration (`phi`) and chain degree of polymerization
    (`nw`).

    Usage:

    >>> generator = SampleGenerator("Bg", batch_size=64, device=torch.device("cuda"))
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
        *,
        parameter: psst.Parameter,
        phi_range: psst.Range,
        nw_range: psst.Range,
        visc_range: psst.Range,
        bg_range: psst.Range,
        bth_range: psst.Range,
        pe_range: psst.Range,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        generator: torch.Generator | None = None,
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

        # Create tensors for phi (concentration) and Nw (number of repeat units per chain)
        # Both are broadcastable to size (batch_size, phi_range.shape[0], nw_range.shape[0])
        # for simple, element-wise operations
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
