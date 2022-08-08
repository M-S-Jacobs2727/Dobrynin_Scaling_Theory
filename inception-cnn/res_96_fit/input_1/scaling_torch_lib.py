import collections
import numpy as np
import torch
import random

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(3e-5, 2e-1)
NW = Param(25, 2.3e5)
ETA_SP = Param(torch.tensor(1), torch.tensor(3.1e6))

#BG = Param(0.29, 1.53)
BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(2.5, 13.5)

#EXP_BG_MU = -0.45297
#EXP_BG_STDEV = 0.41482
#EXP_BTH_MU = -1.00593
#EXP_BTH_STDEV = 0.33205
#EXP_PE_MU = 2.19325
#EXP_PE_STDEV = 0.32861

ETA_SP_131 = Param(torch.tensor(0.1), torch.tensor(1.5e5))
ETA_SP_2 = Param(torch.tensor(1), torch.tensor(4e6))

def unnormalize_B_params(y):

    Bg = y[:,0] * (BG.max - BG.min) + BG.min
    Bth = y[:,1] * (BTH.max - BTH.min) + BTH.min

    return Bg, Bth

def unnormalize_params(y):
    """Simple linear normalization.
    """
    Bg = y[:, 0] * (BG.max - BG.min) + BG.min #+ 1e-4
    Bth = y[:, 1] * (BTH.max - BTH.min) + BTH.min #+ 1e-4
    Pe = y[:, 2] * (PE.max - PE.min) + PE.min #+ 1e-4
    return Bg, Bth, Pe

def unnormalize_params_plot(y):
    """Simple linear normalization.
    """
    Bg = y[:, 0] * (BG.max - BG.min) + BG.min
    Bth = y[:, 1] * (BTH.max - BTH.min) + BTH.min
    Pe = y[:, 2] * (PE.max - PE.min) + PE.min
    return Bg, Bth, Pe

def unnormalize_params2(Bg, Bth, Pe):
    """Simple linear normalization.
    """
    Bg = Bg * (BG.max - BG.min) + BG.min #+ 1e-4
    Bth = Bth * (BTH.max - BTH.min) + BTH.min #+ 1e-4
    Pe = Pe * (PE.max - PE.min) + PE.min #+ 1e-4
    return Bg, Bth, Pe

def unnormalize_Pe(Pe_unnorm):
    return  Pe_unnorm * (PE.max - PE.min) + PE.min 

def normalize_Pe(Pe):
    return (Pe - PE.min) / (PE.max - PE.min)

def normalize_bth(Bth):
    norm_Bth = (Bth - BTH.min) / (BTH.max - BTH.min)
    return norm_Bth

def renormalize_params(y):

    y[:,0] = (y[:,0] - BG.min) / (BG.max - BG.min)
    y[:,1] = (y[:,1] - BTH.min) / (BTH.max - BTH.min)
    y[:,2] = (y[:,2] - PE.min) / (PE.max - PE.min)
    return y

def add_noise(eta_sp: torch.Tensor):
    """Add noise
    """
    eta_sp += eta_sp * 0.05 * torch.normal(
        torch.zeros_like(eta_sp),
        torch.ones_like(eta_sp)
    )

    return eta_sp

def eta_131_norm(eta_sp_131: torch.Tensor):
    """Cap 131 normalization, take log and normalize to [0,1]
    """
    eta_sp_131 = torch.fmin(eta_sp_131, ETA_SP_131.max)
    eta_sp_131 = torch.fmax(eta_sp_131, ETA_SP_131.min)
    return (torch.log(eta_sp_131) - torch.log(ETA_SP_131.min)) / \
        (torch.log(ETA_SP_131.max) - torch.log(ETA_SP_131.min))

def eta_2_norm(eta_sp_2: torch.Tensor):
    """Cap 2 normalization, take log and normalize to [0,1]
    """
    eta_sp_2 = torch.fmin(eta_sp_2, ETA_SP_2.max)
    eta_sp_2 = torch.fmax(eta_sp_2, ETA_SP_2.min)
    return (torch.log(eta_sp_2) - torch.log(ETA_SP_2.min)) / \
        (torch.log(ETA_SP_2.max) - torch.log(ETA_SP_2.min))

def normalize_visc(eta_sp: torch.Tensor):
    """Cap the values, take the log, then normalize.
    """
    eta_sp = torch.fmin(eta_sp, ETA_SP.max)
    eta_sp = torch.fmax(eta_sp, ETA_SP.min)
    return (torch.log(eta_sp) - torch.log(ETA_SP.min)) / \
        (torch.log(ETA_SP.max) - torch.log(ETA_SP.min))

def surface_generator(num_batches: int, batch_size: int,
                      device: torch.device,
                      resolution: 'tuple[int]' = (96, 96)):
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
    ETA_SP_131.min.to(dtype=torch.float, device=device)
    ETA_SP_131.max.to(dtype=torch.float, device=device)
    ETA_SP_2.min.to(dtype=torch.float, device=device)
    ETA_SP_2.max.to(dtype=torch.float, device=device)

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

    def generate_surfaces(Bg, Bth, Pe):
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        g = torch.fmin(
            Bg**(3/0.764) / phi**(1/0.764),
            Bth**6 / phi**2
        )

        g[g < 1] = 0  

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling
        Ne = Pe**2 * g * torch.fmin(
            torch.tensor([1], device=device), torch.fmin(
                (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
                Bth**4 * phi**(2/3)
            )
        )

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover at c = c**
        eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
            1/g,
            phi / Bth**2
        )

        #eta_sp = eta_sp.to(torch.float64)

        eta_sp = add_noise(eta_sp)

        # Nw trim
        for i in range(0,batch_size):
        # make sure that random Nw selector selects a range where all Nw have at least one value of eta_sp > 1
            arr = (eta_sp[i] > 0).nonzero(as_tuple=True)[0].unique()
            arr = arr[torch.sum((eta_sp[i] > ETA_SP.min) & (eta_sp[i] < ETA_SP.max), dim=1) > 1]
            arr_lo = (eta_sp[i] < 20).nonzero(as_tuple=True)[0].unique()
            if len(arr_lo) < 2:
                print(Bg[i,0,0], Bth[i,0,0], Pe[i,0,0])
            trim_lo = torch.randint(arr_lo.min(), arr_lo.max(), (1, 1))
            arr_hi = ((eta_sp[i] > ETA_SP.min) & (eta_sp[i] < ETA_SP.max)).nonzero(as_tuple=True)[0].unique()
            trim_hi = torch.randint(arr_hi.min(), arr_hi.max(), (1, 1))
            trim = torch.tensor((trim_lo, trim_hi))
            arr2 = arr[(arr < trim.min()) | (arr > trim.max())]
            #trim = torch.randint(arr_low.min(), arr.max(), (1,2)).sort(1)[0]   
            #arr2 = arr[(arr < trim[0][0]) | (arr > trim[0][1])] 
            eta_sp[i, arr2] = 0
        eta_sp[eta_sp < ETA_SP.min] = 0
        eta_sp[eta_sp > ETA_SP.max] = 0
        eta_sp[eta_sp.isnan()] = 0

        # return range of eta_sp values. All eta_sp values simillar to how experimental datasets should look
        # eta_sp < 1 and > 3.1e6 are set to zero
        return eta_sp

    for _ in range(num_batches):

        Bg = torch.rand(batch_size).to(device)*(BG.max-BG.min)+BG.min
        Bth = torch.rand(batch_size).to(device)*(BTH.max-BTH.min)+BTH.min
        #Bg = torch.empty(batch_size).log_normal_(mean=EXP_BG_MU, std=EXP_BG_STDEV).to(device)+0.216
        #Bth = torch.empty(batch_size).log_normal_(mean=EXP_BTH_MU, std=EXP_BTH_STDEV).to(device)+0.216
        #Pe = torch.empty(batch_size).log_normal_(mean=EXP_PE_MU, std=EXP_PE_STDEV).to(device)
        Pe = torch.rand(batch_size).to(device)*(PE.max-PE.min)+PE.min
       

        # make sure Bg values lie between BGmin and BGmax of exp dataset (0.389, 1.53)
        #Bg = torch.where(Bg > BG.max, random.uniform(0,BG.max-0.389)+0.389, Bg).to(device)
        # make sure Bth values lie between 0 and BG** (1/0.824)
        #Bth = torch.where(Bth > Bg ** (1 / 0.824), random.uniform(0,1)*(Bg ** (1 / 0.824)), Bth).to(device)

        Bth = torch.where(Bth > Bg ** (1 / 0.824), torch.rand(1).to(device) * (Bg ** (1 / 0.824) - BTH.min) + BTH.min, Bth).to(device)
        Pe = torch.where(Pe > PE.max, torch.rand(1).to(device)*PE.max, Pe).to(device)
        Pe = torch.where(Pe < 3, Pe + 3, Pe).to(device)

        y = torch.column_stack((Bg, Bth, Pe)).to(device)
        eta_sp = generate_surfaces(Bg, Bth, Pe)
        #eta_sp = add_noise(eta_sp)

        eta_sp_131 = eta_sp/Nw/phi**(1/(3*0.588-1))
        eta_sp_2 = eta_sp/Nw/phi**2
        X = normalize_visc(eta_sp).to(torch.float)
        X_131 = eta_131_norm(eta_sp_131).to(torch.float)
        X_2 = eta_2_norm(eta_sp_2).to(torch.float)
        y = renormalize_params(y)

        #y2 = y[:,0:2]

        #X_test = torch.stack((X, X_131, X_2), dim=1)
        #print(X_test.size())
        #print(X.size())
        X_test = torch.unsqueeze(X,1)
        #X_test = torch.stack((X), dim=1)
        yield X_test, y, eta_sp

def main():
    """For testing only.
    """
    for surf in voxel_image_generator(8, 1, torch.device('cpu')):
        X, y = surf


if __name__ == '__main__':
    main()
