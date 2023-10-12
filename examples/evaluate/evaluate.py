from functools import partial
from pathlib import Path

import numpy as np
import torch

from ...core.surface_generator import normalize
from ...core.configuration import GeneratorConfig

def exp_data_to_images(
        infile: str | Path,
        outfile: str | Path,
        config: GeneratorConfig,
        force: bool = False
):
    outfile = Path(outfile)
    if not force and outfile.is_file():
        return torch.load(outfile)
    
    infile = Path(infile)
    data = np.loadtxt(infile, usecols=tuple(range(3, 10)), skiprows=1, delimiter=',')
    systems: list[torch.Tensor] = list()
    for group in np.unique(data[:, 0]):
        systems.append(torch.as_tensor(data[data[:, 0] == group][:, 1:]))
    
    Bg = torch.zeros(len(systems))
    Bth = torch.zeros_like(Bg)
    Pe = torch.zeros_like(Bg)
    phi = config.phi_range.flatten()
    nw = config.nw_range.flatten()
    visc = torch.zeros((len(systems), phi.shape[0], nw.shape[0]))

    phi_bins = torch.linspace(0, 1, phi.shape[0])
    nw_bins = torch.linspace(0, 1, nw.shape[0])

    def get_indices(raw_data: torch.Tensor, min: float, max: float, bins: torch.Tensor):
        norm_values = normalize(raw_data, min, max, log_scale=True)
        indices = torch.argmin(torch.abs(norm_values.reshape(-1, 1) - bins.reshape(1, -1)), dim=1)
        return indices
    
    get_phi_indices = partial(get_indices, min=phi.min(), max=phi.max(), bins=phi_bins)
    get_nw_indices = partial(get_indices, min=nw.min(), max=nw.max(), bins=nw_bins)

    for i, system in enumerate(systems):
        Bg[i] = system[0, 0]
        Bth[i] = system[0, 1]
        Pe[i] = system[0, 2]

        phi_idx = get_phi_indices(system[:, 3])
        nw_idx = get_nw_indices(system[:, 4])

        visc[i, phi_idx, nw_idx] = system[:, 5]
    
    torch.save((Bg, Bth, Pe, phi, nw, visc), infile.with_suffix(".pt"))

    return Bg, Bth, Pe, phi, nw, visc
