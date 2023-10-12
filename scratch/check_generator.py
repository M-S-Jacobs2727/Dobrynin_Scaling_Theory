import collections

import numpy as np
import torch

device = torch.device("cpu")
Param = collections.namedtuple("Param", ("min", "max"))
batch_size = 64

PHI = Param(3e-5, 2e-2)
NW = Param(100, 1e5)
ETA_SP = Param(torch.tensor(1), torch.tensor(1e6))

BG_COMBO = Param(0.36, 1.55)
BTH_COMBO = Param(0.22, 0.82)
PE_COMBO = Param(3.2, 13.5)

BG_ATH = Param(0.36, 1.25)
PE_ATH = Param(2.5, 6.5)

BTH_TH = Param(0.25, 0.5)
PE_TH = Param(6, 10)

# theoretical limits based on ETA_SP:
ETA_SP_131 = Param(
    ETA_SP.min / NW.max / PHI.max ** (1 / (3 * 0.588 - 1)),
    ETA_SP.max / NW.min / PHI.min ** (1 / (3 * 0.588 - 1)),
)
ETA_SP_2 = Param(ETA_SP.min / NW.max / PHI.max**2, ETA_SP.max / NW.min / PHI.min**2)


def ryan_viscosity(
    Bg: torch.Tensor,
    Bth: torch.Tensor,
    Pe: torch.Tensor,
    phi: torch.Tensor,
    nw: torch.Tensor,
) -> torch.Tensor:
    # Number of repeat units per correlation blob
    # Only defined for c < c**
    # Minimum accounts for crossover at c = c_th
    g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth**6 / phi**2)

    g[g < 1] = 0

    phi_th = Bth**3 * (Bth / Bg) ** (1 / (2 * 0.588 - 1))
    Bth6 = Bth**6
    phi_star_star = Bth**4
    param = (Bth / Bg) ** (1 / (2 * 0.588 - 1)) / Bth**3

    lam_g_KN = torch.where(phi < phi_star_star, 1, (phi / phi_star_star) ** (-3 / 2))
    lam_g_RC = torch.where(
        phi < phi_th,
        phi_th ** (2 / 3) * Bth**-4,
        torch.where(
            phi < Bth6,
            phi ** (2 / 3) * Bth**-4,
            torch.where(phi < phi_star_star, 1, (phi / phi_star_star) ** (-3 / 2)),
        ),
    )

    lam_g = torch.where(param > 1, lam_g_KN, lam_g_RC)

    Ne = Pe**2 * g * lam_g

    eta_sp = nw * (1 + (nw / Ne) ** 2) * torch.fmin(1 / g, phi / Bth**2)

    return eta_sp


def mike_viscosity(
    Bg: torch.Tensor,
    Bth: torch.Tensor,
    Pe: torch.Tensor,
    phi: torch.Tensor,
    nw: torch.Tensor,
) -> torch.Tensor:
    g = torch.fmin(
        (Bg**3 / phi) ** (1 / 0.764),
        Bth**6 / phi**2,
    )

    # Number of repeat units per entanglement strand
    # Universal definition of Ne accounts for both
    # Kavassalis-Noolandi and Rubinstein-Colby scaling.
    # Corresponding concentration ranges are listed next to the expressions.
    Ne = Pe**2 * torch.fmin(
        torch.fmin(
            Bth ** (0.944 / 0.528) / Bg ** (2 / 0.528) * g,  # c* < c < c_th
            Bth**2 * phi ** (-4 / 3),  # c_th < c < b^-3
        ),
        g,  # b^-3 < c
    )

    eta_sp = nw * (1 + (nw / Ne) ** 2) * torch.fmin(1 / g, phi / Bth**2)

    return eta_sp


def main():
    phi = torch.tensor(
        np.geomspace(PHI.min, PHI.max, 224, endpoint=True),
        dtype=torch.float,
        device=device,
    ).reshape(1, 224, 1)

    nw = torch.tensor(
        np.geomspace(NW.min, NW.max, 224, endpoint=True),
        dtype=torch.float,
        device=device,
    ).reshape(1, 1, 224)

    r = torch.rand(size=(batch_size, 1, 1), device=device)
    Bg: torch.Tensor = r * (BG_COMBO.max - BG_COMBO.min) + BG_COMBO.min
    Bg = Bg.reshape(batch_size, 1, 1)

    r = torch.rand(size=(batch_size, 1, 1), device=device)
    Bth: torch.Tensor = r * (BTH_COMBO.max - BTH_COMBO.min) + BTH_COMBO.min
    Bth = Bth.reshape(batch_size, 1, 1)

    r = torch.rand(size=(batch_size, 1, 1), device=device)
    Pe: torch.Tensor = r * (PE_COMBO.max - PE_COMBO.min) + PE_COMBO.min
    Pe = Pe.reshape(batch_size, 1, 1)

    ryan = ryan_viscosity(Bg, Bth, Pe, phi, nw)
    mike = mike_viscosity(Bg, Bth, Pe, phi, nw)

    diff = ryan - mike

    print(diff.min())
    print(diff.max())


if __name__ == "__main__":
    main()
