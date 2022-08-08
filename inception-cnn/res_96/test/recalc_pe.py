import collections
import numpy as np
import torch
import random
import torch
from scipy.optimize import curve_fit

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(1e-6, 1e-2)
NW = Param(25, 2.3e5)
ETA_SP = Param(torch.tensor(1), torch.tensor(3.1e6))

BG = Param(0, 1.53)
BTH = Param(0, 0.8)
PE = Param(0, 13.3)

EXP_BG_MU = -0.4601
EXP_BG_STDEV = 0.30511
EXP_BTH_MU = -1.02699
EXP_BTH_STDEV = 0.24407
EXP_PE_MU = 1.76267
EXP_PE_STDEV = 0.42684

ETA_SP_131 = Param(torch.tensor(0.1), torch.tensor(1e5))
ETA_SP_2 = Param(torch.tensor(1), torch.tensor(4e6))

class linearRegression(torch.nn.Module):

    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inpitSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

def pe_fit_function(x, Pe):

    return Pe**2 * x

def recalc_pe_prediction(device, Bg, Bth, eta_sp, resolution, batch_size):

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
    phi = torch.where(eta_sp == 0, 0, phi)
    Nw = torch.where(eta_sp == 0, 0, Nw)
    #phi_tile = torch.tile(phi, (batch_size, 1, 1))
    #Nw_tile = torch.tile(Nw, (batch_size, 1, 1))
    shape = torch.Size((1, *(phi.size()[1:])))
    Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
    Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)

    g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth ** 6 / phi ** 2) 
    Ne = Nw * (eta_sp / torch.fmin(1 / g, phi / Bth ** 2) / Nw - 1) ** (-0.5)

    lam_g_g = torch.fmin(torch.fmin(g*phi**(2/3)/Bth**4, Bth**2*phi**(4/3)),g)

    popt_arr = torch.tensor(np.zeros(shape=(batch_size,1)))

    for i in range(0, batch_size):

        Ne_slice = Ne[i][eta_sp[i] != 0]
        lam_g_g_slice = lam_g_g[i][eta_sp[i] != 0]
        Ne_filter = Ne_slice[(~Ne_slice.isnan()) & (~lam_g_g_slice.isnan()) & (~Ne_slice.isinf()) & (~lam_g_g_slice.isinf())]
        lam_g_g_filter = lam_g_g[i][eta_sp[i] != 0][(~Ne_slice.isnan()) & (~lam_g_g_slice.isnan()) & (~Ne_slice.isinf()) & (~lam_g_g_slice.isinf())]

        y_data = Ne_filter.detach().cpu().numpy()
        x_data = lam_g_g_filter.detach().cpu().numpy()

        if not np.any(y_data):
            val = 0
        else:

            popt, pcov = curve_fit(pe_fit_function, x_data, y_data)
            val = popt[0]
            popt_arr[i] = val
        #model = linearRegression()
        #criterion = torch.nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    return popt_arr
