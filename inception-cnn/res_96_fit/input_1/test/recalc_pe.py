import collections
import numpy as np
import torch
import random
import torch
from scipy.optimize import curve_fit
from lmfit.models import StepModel, LinearModel

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

def pe_fit_function(x, Pe):

    return Pe**2 * x

def recalc_pe_prediction(device, Bg, Bth, eta_sp, resolution, batch_size):

    #torch.set_printoptions(precision=8)   

    phi = torch.tensor(np.geomspace(
        PHI.min,
        PHI.max,
        resolution[0],
        endpoint=True
    ), dtype=torch.float64, device=device)

    Nw = torch.tensor(np.geomspace(
        NW.min,
        NW.max,
        resolution[1],
        endpoint=True
    ), dtype=torch.float64, device=device)

    phi, Nw = torch.meshgrid(phi, Nw, indexing='xy')

    shape = torch.Size((1, *(phi.size()[1:])))
    Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
    Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)

    g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth ** 6 / phi ** 2)

    lam_g = torch.fmin(
            torch.tensor([1], device=device), torch.fmin(
                (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
                Bth**4 * phi**(2/3)
            )
        )

    phi_star_star = Bth**4

    lam = 1 / lam_g * torch.where(phi < phi_star_star, 1, (phi / phi_star_star) ** 0.5)

    lam_g_g = lam_g * g

    Ne = Nw * (lam_g_g * lam * eta_sp / Nw - 1) ** -0.5  

    popt_arr = torch.tensor(np.zeros(shape=(batch_size,1)))

    for i in range(0, batch_size):

        x = lam_g_g[i][~Ne[i].isnan()]
        y = Ne[i][~Ne[i].isnan()]

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()


        if len(x) == 0:
            popt_arr[i] = 0

        else:
            popt, pcov = curve_fit(pe_fit_function, x, y)
            popt_arr[i] = popt[0]

        
        # debug #
        #Pe_true = Pe_true.detach().cpu().numpy()
        #plt.scatter(x,y)
        #plt.plot(x, pe_fit_function(x, popt[0]), color='black')
        #plt.plot(x, pe_fit_function(x, Pe[i].detach().cpu().numpy()), color='red')
        #plt.xscale("log")
        #plt.yscale("log")
        #print(popt[0], Pe[i])
        #print(popt[0], Pe_true[i])
        #plt.show()



    return popt_arr
