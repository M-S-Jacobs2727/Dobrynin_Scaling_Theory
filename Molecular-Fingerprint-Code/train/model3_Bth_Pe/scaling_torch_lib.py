import collections
import numpy as np
import torch

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(1e-6, 1e-2)
NW = Param(10, 3e5)
ETA_SP = Param(torch.tensor(1), torch.tensor(3.5e6))

#BG = Param(0, 1.53)
BTH = Param(0, 0.8)
PE = Param(0, 13.3)


#EXP_BG_MU = -0.4601
#EXP_BG_STDEV = 0.30511
EXP_BTH_MU = -1.02699
EXP_BTH_STDEV = 0.24407
EXP_PE_MU = 1.76267
EXP_PE_STDEV = 0.42684

def unnormalize_params(y):
    """Simple linear normalization.
    """
    #Bg = y[:, 0] * (BG.max - BG.min) + BG.min + 1e-4
    Bth = y[:, 0] * (BTH.max - BTH.min) + BTH.min + 1e-4
    Pe = y[:, 1] * (PE.max - PE.min) + PE.min + 1e-4
    #return Bg, Bth, Pe

    return Bth, Pe

def unnormalize_params2(Bg, Pe):
    """Simple linear normalization.
    """
    Bg = Bg * (BG.max - BG.min) + BG.min + 1e-4
    #Bth = Bth * (BTH.max - BTH.min) + BTH.min + 1e-4
    Pe = Pe * (PE.max - PE.min) + PE.min + 1e-4
    return Bg, Bth, Pe

def unnormalize_Pe(Pe_unnorm):
    return  Pe_unnorm * (PE.max - PE.min) + PE.min + 1e-4

def normalize_bth(Bth):
    norm_Bth = (Bth - BTH.min) / (BTH.max - BTH.min)
    return norm_Bth

def renormalize_params(y):

    #y[:,0] = (y[:,0] - BG.min) / (BG.max - BG.min)
    y[:,0] = (y[:,0] - BTH.min) / (BTH.max - BTH.min)
    y[:,1] = (y[:,1] - PE.min) / (PE.max - PE.min)
    return y

def normalize_visc(eta_sp: torch.Tensor):
    """Add noise, cap the values, take the log, then normalize.
    """
    eta_sp += eta_sp * 0.05 * torch.normal(
        torch.zeros_like(eta_sp),
        torch.ones_like(eta_sp)
    )
    eta_sp = torch.fmin(eta_sp, ETA_SP.max)
    eta_sp = torch.fmax(eta_sp, ETA_SP.min)
    return (torch.log(eta_sp) - torch.log(ETA_SP.min)) / \
        (torch.log(ETA_SP.max) - torch.log(ETA_SP.min))

def surface_generator(num_batches: int, batch_size: int,
                      device: torch.device,
                      resolution: 'tuple[int]' = (32, 32)):
    """Generate `batch_size` surfaces, based on ranges for `Bg`, `Bth`, and
    `Pe`, to be used in a `for` loop.

    It defines the resolution of the surface based on either user input
    (keyword argument `resolution`). It then generates random values for `Bg`,
    `Bth`, and `Pe`, evaluates the `(phi, Nw, eta_sp)` surface, and normalizes
    the result. The normalized values of `eta_sp` and `(Bg, Bth, Pe)` are
    yielded as `X` and `y` for use in a neural network.

    Input:
        `num_batches` (`int`) : The number of loops to be iterated through.
        `batch_size` (`int`) : The length of the generated values.
        `device` (`torch.device`): The device to do computations on.
        `resolution` (tuple of `int`s) : The shape of the last two dimensions
            of the generated values.

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Generated,
            normalized values of `eta_sp` at indexed `phi` and `Nw`.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """

    ETA_SP.min.to(dtype=torch.float, device=device)
    ETA_SP.max.to(dtype=torch.float, device=device)

    # Create tensors for phi (concentration) and Nw (chain length)
    # Both are meshed and tiled to cover a 3D tensor of size
    # (batch_size, *resolution) for simple, element-wise operations
    phi = torch.tensor(np.geomspace(
        PHI.min,
        PHI.max,
        resolution[0],
        endpoint=True
    ), dtype=torch.float, device=device)

    Nw = torch.tensor(np.geomspace(
        NW.min,
        NW.max,
        resolution[1],
        endpoint=True
    ), dtype=torch.float, device=device)

    phi, Nw = torch.meshgrid(phi, Nw, indexing='xy')
    phi = torch.tile(phi, (batch_size, 1, 1))
    Nw = torch.tile(Nw, (batch_size, 1, 1))

    def generate_surfaces(Bth, Pe):
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        #Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        #g = torch.fmin(
        #    Bg**(3/0.764) / phi**(1/0.764),
        #    Bth**6 / phi**2
        #)

        # g for athermal #
        #g = Bg**(3/0.764) / phi**(1/0.764)
        # end #

        # g for thermal blob #
        g = Bth**6 / phi**2

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling
        #Ne = Pe**2 * g * torch.fmin(
        #    torch.tensor([1], device=device), torch.fmin(
        #        (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
        #        Bth**4 * phi**(2/3)
        #    )
        #)

        # Ne for athermal #
        Ne = Pe**2 * g

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover at c = c**
        eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
            1/g,
            phi / Bth**2
        )

        # case - no Bth #
        #eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
        #        1/g,
        #        phi / Bg ** (1/0.412)
        #        )

        return eta_sp

    for _ in range(num_batches):

        #Bg = torch.empty(batch_size).log_normal_(mean=EXP_BG_MU, std=EXP_BG_STDEV).to(device)
        Bth = torch.empty(batch_size).log_normal_(mean=EXP_BTH_MU, std=EXP_BTH_STDEV).to(device)
        Pe = torch.empty(batch_size).log_normal_(mean=EXP_PE_MU, std=EXP_PE_STDEV).to(device)

        y = torch.column_stack((Bth, Pe)).to(device)
        eta_sp = generate_surfaces(Bth, Pe)
        y = renormalize_params(y)
        X = normalize_visc(eta_sp).to(torch.float)
        yield X, y

#def main():
#    """For testing only.
#    """
#    for surf in voxel_image_generator(8, 1, torch.device('cpu')):
#        X, y = surf
#
#
#if __name__ == '__main__':
#    main()
