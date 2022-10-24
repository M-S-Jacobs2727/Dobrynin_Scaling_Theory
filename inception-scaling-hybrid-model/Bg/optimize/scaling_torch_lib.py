import collections
import numpy as np
import torch
import random

Param = collections.namedtuple('Param', ('min', 'max'))

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
ETA_SP_131 = Param(ETA_SP.min/NW.max/PHI.max**(1/(3*0.588-1)),ETA_SP.max/NW.min/PHI.min**(1/(3*0.588-1)))
ETA_SP_2 = Param(ETA_SP.min/NW.max/PHI.max**2,ETA_SP.max/NW.min/PHI.min**2)

def unnormalize_B_params(y):

    Bg = y[:,0] * (BG.max - BG.min) + BG.min
    Bth = y[:,1] * (BTH.max - BTH.min) + BTH.min

    return Bg, Bth

def unnormalize_Bg_param(y):

    Bg = y[:,0] * (BG_COMBO.max - BG_COMBO.min) + BG_COMBO.min

    return Bg

def unnormalize_Bth_param(y):

    Bth = y[:,0] * (BTH_COMBO.max - BTH_COMBO.min) + BTH_COMBO.min

    return Bth

def renormalize_params(y):

    y[:,0] = (y[:,0] - BG_COMBO.min) / (BG_COMBO.max - BG_COMBO.min)
    y[:,1] = (y[:,1] - BTH_COMBO.min) / (BTH_COMBO.max - BTH_COMBO.min)
    y[:,2] = (y[:,2] - PE_COMBO.min) / (PE_COMBO.max - PE_COMBO.min)
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

def surface_generator_Bth(num_batches: int, batch_size: int,
                      device: torch.device,
                      return_nw: bool,
                      resolution: 'tuple[int]' = (224, 224)):
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

    def generate_surfaces(Bg, Bth, Pe, a):
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)
        a_tile = torch.tile(a.reshape((batch_size, 1, 1)), shape)
        lam_g = torch.tile(a.reshape((batch_size, 1, 1)), shape)

        Bg[Bg==0] = torch.nan

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        g = torch.fmin(
            Bg**(3/0.764) / phi**(1/0.764),
            Bth**6 / phi**2
        )

        g[g < 1] = 0  

        phi_star = Bg**3*Nw**(1-3*0.588)
        phi_th = Bth**3*(Bth/Bg)**(1/(2*0.588-1))
        Bth6 = Bth**6
        phi_star_star = Bth**4
        param = (Bth/Bg)**(1/(2*0.588-1))/Bth**3

        lam_g_KN = torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))
        lam_g_RC = torch.where(phi<phi_th, phi_th**(2/3)*Bth**-4, torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))))
        lam_g_Bth = torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2)))
        lam_KN = 1 / lam_g_KN * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        lam_RC = 1 / lam_g_RC * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        lam_Bth = 1 / lam_g_Bth * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))

        lam_g = torch.where(a_tile == 1, 1, lam_g_Bth)
        lam_g = torch.where(a_tile == 0, torch.where(param > 1, lam_g_KN, lam_g_RC), lam_g)

        lam = 1 / lam_g * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        lam = torch.where(lam.isnan(), 1, lam)

        Ne = Pe**2 * g * lam_g

        eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
            1/g,
            phi / Bth**2
        )

        eta_sp = add_noise(eta_sp)
        eta_sp[eta_sp > ETA_SP.max] = 0
        eta_sp[eta_sp < ETA_SP.min] = 0
        eta_sp[eta_sp.isnan()] = 0
        eta_sp[eta_sp.isinf()] = 0
        eta_sp[phi < phi_star] = 0

        # Nw trim

        if return_nw:
            Nw_used = {}
            Nw_min = np.zeros(batch_size)
            Nw_max = np.zeros(batch_size)
            Num_Nw = np.zeros(batch_size)

        eta_sp_copy = eta_sp.clone()

        for i in range(0,batch_size):
            # declare eta_sp cutoff values
            eta_sp_lo_cutoff = 2.0
            eta_sp_hi_cutoff = 10.0

            # make sure that random Nw selector selects a range where all Nw have at least one value of eta_sp > 1
            arr_all = torch.arange(0, 224).to(device)

            choose_num = 100

            while len(arr_all[torch.sum(eta_sp[i] > 0, dim=1) > choose_num]) < 20:
                choose_num -= 1

            arr_choose = arr_all[torch.sum(eta_sp[i] > 0, dim=1) > choose_num]

            num_nw = torch.randint(1, 16, (1,1))[0][0]
            trim_nw_arr = torch.randint(arr_choose.min(), arr_choose.max(), (1, num_nw)).unique().to(device)

            
            if return_nw:
                Nw_used[i] = Nw[0, :, 0][trim_nw_arr].detach().cpu().numpy()
                Nw_min = np.min(Nw[0, :, 0][trim_nw_arr].detach().cpu().numpy())
                Nw_max = np.max(Nw[0, :, 0][trim_nw_arr].detach().cpu().numpy())

            trim_zeros_nw = ~torch.isin(arr_all, trim_nw_arr.to(device))

            eta_sp[i, trim_zeros_nw] = 0

            tensor_phi_arr = (eta_sp[i] > 0).nonzero(as_tuple=True)[1].unique()

            trim_phi_arr = torch.randint(tensor_phi_arr.min(), tensor_phi_arr.max(), (1, 16)).unique().to(device)

            trim_zeros = ~torch.isin(arr_all, trim_phi_arr.to(device))     

            eta_sp[i, :, trim_zeros] = 0 

        # return range of eta_sp values. All eta_sp values simillar to how experimental datasets should look
        # eta_sp < 1 and > 3.1e6 are set to zero

        if return_nw:
            return eta_sp, Nw_used, Nw_min, Num_Nw
        else:
            return eta_sp

    for _ in range(num_batches):

        # Declare labels for a
        # label 0 = both Bg and Bth
        # label 1 = only Bth
        a = torch.randint(low=0, high=2, size=(batch_size, 1)).to(device)
        Bg_Bth_Pe_vals = torch.zeros(batch_size, 3).to(device)
        Pe_vals = torch.where(a == 0, torch.rand(batch_size).to(device)*(PE_COMBO.max - PE_COMBO.min) + PE_COMBO.min,0).diagonal().to(device)

        Pe_1_vals = torch.where(a == 1, torch.rand(batch_size).to(device)*(PE_TH.max - PE_TH.min) + PE_TH.min,Pe_vals).diagonal().to(device)

        Bg_diag = torch.where(a == 0, torch.rand(batch_size).to(device)*(BG_COMBO.max-BG_COMBO.min)+BG_COMBO.min, 0).diagonal().to(device)
        Bth_diag = torch.where(a == 0, torch.rand(batch_size).to(device)*(BTH_COMBO.max-BTH_COMBO.min)+BTH_COMBO.min, 0).diagonal().to(device)
        Bth_diag = torch.where(Bth_diag > Bg_diag ** (1 / 0.824), torch.rand(1).to(device) * (Bg_diag ** (1 / 0.824) - BTH_COMBO.min) + BTH_COMBO.min, Bth_diag).to(device)

        Bth_model_vals_diag = torch.where(a == 1, torch.rand(batch_size).to(device)*(BTH_TH.max-BTH_TH.min)+BTH_TH.min, 0).diagonal().to(device)

        Bth_diag = torch.where(Bth_diag == 0, Bth_model_vals_diag, Bth_diag).to(device)
        Bg_Bth_Pe_vals[:,0] = Bg_diag
        Bg_Bth_Pe_vals[:,1] = Bth_diag
        Bg_Bth_Pe_vals[:,2] = Pe_1_vals

        Bg = Bg_Bth_Pe_vals[:, 0]
        Bth = Bg_Bth_Pe_vals[:, 1]
        Pe = Bg_Bth_Pe_vals[:, 2]

        y = torch.column_stack((Bg, Bth, Pe)).to(device)

        # return Nw?
        # if Nw_bool = True
        # else

        if return_nw:
            eta_sp, Nw_used, Nw_min, Num_Nw = generate_surfaces_Bth(Bg, Bth, Pe, a)
        else:
            eta_sp = generate_surfaces(Bg, Bth, Pe, a)

        eta_sp_2 = eta_sp/Nw/phi**2
        X = normalize_visc(eta_sp).to(torch.float)
        X_2 = eta_2_norm(eta_sp_2).to(torch.float)
        y = renormalize_params(y)

        X_2 = torch.unsqueeze(X_2, 1)
        y_Bth = y[:,1].unsqueeze(1)

        if return_nw:
            Nw_min.reshape(-1,1).T
            Nw_max.reshape(-1,1).T
            Num_Nw.reshape(-1,1).T
            yield X_2, y_Bth, Nw_min, Nw_max, Num_Nw

        else:
            yield X_2, y_Bth

def surface_generator_Bg(num_batches: int, batch_size: int,
                      device: torch.device,
                      return_nw: bool,
                      resolution: 'tuple[int]' = (224, 224)):
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

    def generate_surfaces(Bg, Bth, Pe, a):
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)
        a_tile = torch.tile(a.reshape((batch_size, 1, 1)), shape)
        lam_g = torch.tile(a.reshape((batch_size, 1, 1)), shape)

        Bth[Bth==0] = torch.nan

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        g = torch.fmin(
            Bg**(3/0.764) / phi**(1/0.764),
            Bth**6 / phi**2
        )

        g[g < 1] = 0

        phi_star = Bg**3*Nw**(1-3*0.588)
        phi_th = Bth**3*(Bth/Bg)**(1/(2*0.588-1))
        Bth6 = Bth**6
        phi_star_star = Bth**4
        param = (Bth/Bg)**(1/(2*0.588-1))/Bth**3

        lam_g_KN = torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))
        lam_g_RC = torch.where(phi<phi_th, phi_th**(2/3)*Bth**-4, torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))))
        lam_KN = 1 / lam_g_KN * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        lam_RC = 1 / lam_g_RC * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))

        lam_g = torch.where(a_tile == 1, 1, lam_g)
        lam_g = torch.where(a_tile == 0, torch.where(param > 1, lam_g_KN, lam_g_RC), lam_g)

        lam = 1 / lam_g * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        lam = torch.where(lam.isnan(), 1, lam)

        Ne = Pe**2 * g * lam_g

        eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
            1/g,
            phi / Bth**2
        )

        eta_sp = add_noise(eta_sp)
        eta_sp[eta_sp > ETA_SP.max] = 0
        eta_sp[eta_sp < ETA_SP.min] = 0
        eta_sp[eta_sp.isnan()] = 0
        eta_sp[eta_sp.isinf()] = 0
        eta_sp[phi < phi_star] = 0

        # Nw trim

        if return_nw:
            Nw_used = {}
            Nw_min = np.zeros(batch_size)
            Nw_max = np.zeros(batch_size)
            Num_Nw = np.zeros(batch_size)

        eta_sp_copy = eta_sp.clone()

        for i in range(0,batch_size):
            # declare eta_sp cutoff values
            eta_sp_lo_cutoff = 2.0
            eta_sp_hi_cutoff = 10.0

            # make sure that random Nw selector selects a range where all Nw have at least one value of eta_sp > 1
            arr_all = torch.arange(0, 224).to(device)

            choose_num = 100

            while len(arr_all[torch.sum(eta_sp[i] > 0, dim=1) > choose_num]) < 20:
                choose_num -= 1

            arr_choose = arr_all[torch.sum(eta_sp[i] > 0, dim=1) > choose_num]

            num_nw = torch.randint(1, 16, (1,1))[0][0]
            trim_nw_arr = torch.randint(arr_choose.min(), arr_choose.max(), (1, num_nw)).unique().to(device)

            if return_nw:
                Nw_used[i] = Nw[0, :, 0][trim_nw_arr].detach().cpu().numpy()
                Nw_min = np.min(Nw[0, :, 0][trim_nw_arr].detach().cpu().numpy())
                Nw_max = np.max(Nw[0, :, 0][trim_nw_arr].detach().cpu().numpy())

            trim_zeros_nw = ~torch.isin(arr_all, trim_nw_arr.to(device))

            eta_sp[i, trim_zeros_nw] = 0

            tensor_phi_arr = (eta_sp[i] > 0).nonzero(as_tuple=True)[1].unique()

            trim_phi_arr = torch.randint(tensor_phi_arr.min(), tensor_phi_arr.max(), (1, 16)).unique().to(device)

            trim_zeros = ~torch.isin(arr_all, trim_phi_arr.to(device))

            eta_sp[i, :, trim_zeros] = 0

        # return range of eta_sp values. All eta_sp values simillar to how experimental datasets should look
        # eta_sp < 1 and > 3.1e6 are set to zero
        #return eta_sp, lam_KN, lam_RC, Nw_min, Nw_max, Num_Nw

        if return_nw:
            return eta_sp, Nw_min, Nw_max, Num_Nw
        else:
            return eta_sp

    for _ in range(num_batches):

        # Declare labels for a
        # label 0 = both Bg and Bth
        # label 1 = only Bg
        a = torch.randint(low=0, high=2, size=(batch_size, 1)).to(device)
        Bg_Bth_Pe_vals = torch.zeros(batch_size, 3).to(device)
        Pe_vals = torch.where(a == 0, torch.rand(batch_size).to(device)*(PE_COMBO.max - PE_COMBO.min) + PE_COMBO.min,0).diagonal().to(device)

        Pe_1_vals = torch.where(a == 1, torch.rand(batch_size).to(device)*(PE_ATH.max - PE_ATH.min) + PE_ATH.min,Pe_vals).diagonal().to(device)

        Bg_diag = torch.where(a == 0, torch.rand(batch_size).to(device)*(BG_COMBO.max-BG_COMBO.min)+BG_COMBO.min, 0).diagonal().to(device)
        Bth_diag = torch.where(a == 0, torch.rand(batch_size).to(device)*(BTH_COMBO.max-BTH_COMBO.min)+BTH_COMBO.min, 0).diagonal().to(device)
        Bth_diag = torch.where(Bth_diag > Bg_diag ** (1 / 0.824), torch.rand(1).to(device) * (Bg_diag ** (1 / 0.824) - BTH_COMBO.min) + BTH_COMBO.min, Bth_diag).to(device)

        Bg_model_vals_diag = torch.where(a == 1, torch.rand(batch_size).to(device)*(BG_ATH.max-BG_ATH.min)+BG_ATH.min, 0).diagonal().to(device)

        Bg_diag = torch.where(Bg_diag == 0, Bg_model_vals_diag, Bg_diag).to(device)
        Bg_Bth_Pe_vals[:,0] = Bg_diag
        Bg_Bth_Pe_vals[:,1] = Bth_diag
        Bg_Bth_Pe_vals[:,2] = Pe_1_vals

        Bg = Bg_Bth_Pe_vals[:, 0]
        Bth = Bg_Bth_Pe_vals[:, 1]
        Pe = Bg_Bth_Pe_vals[:, 2]

        y = torch.column_stack((Bg, Bth, Pe)).to(device)

        if return_nw:
            eta_sp, Nw_min, Nw_max, Num_Nw = generate_surfaces(Bg, Bth, Pw, a)
        else:
            eta_sp = generate_surfaces(Bg, Bth, Pe, a)

        eta_sp_131 = eta_sp/Nw/phi**(1/(3*0.588-1))
        X = normalize_visc(eta_sp).to(torch.float)
        X_131 = eta_131_norm(eta_sp_131).to(torch.float)
        y = renormalize_params(y)

        X_131 = torch.unsqueeze(X_131, 1)
        y_Bg = y[:,0].unsqueeze(1)

        if return_nw:
            Nw_min.reshape(-1,1).T
            Nw_max.reshape(-1,1).T
            Num_Nw.reshape(-1,1).T
            yield X_131, y_Bg, Nw_min, Nw_max, Num_Nw
        else:
            yield X_131, y_Bg