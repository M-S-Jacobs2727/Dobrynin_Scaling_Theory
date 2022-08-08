import collections
import numpy as np
import torch
import random
import torch
from scipy.optimize import curve_fit
from lmfit.models import StepModel, LinearModel

Param = collections.namedtuple('Param', ('min', 'max'))

#PHI = Param(1e-6, 1e-2)
PHI = Param(3e-5, 1e-2)
NW = Param(25, 2.3e5)
ETA_SP = Param(torch.tensor(1), torch.tensor(3.1e6))

BG = Param(0.282, 1.53)
BTH = Param(0.216, 0.8)
#BG = Param(0, 1.53)
#BTH = Param(0, 0.8)
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

    #eta_sp = torch.where((eta_sp < 1) &, 0, eta_sp)
    phi = torch.where(eta_sp == 0, 0, phi)
    Nw = torch.where(eta_sp ==0, 0, Nw)
    #phi_tile = torch.tile(phi, (batch_size, 1, 1))
    #Nw_tile = torch.tile(Nw, (batch_size, 1, 1))
    shape = torch.Size((1, *(phi.size()[1:])))
    Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
    Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)

    Bg = torch.where(eta_sp == 0, 0, Bg)
    Bth = torch.where(eta_sp == 0, 0, Bth)

    g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth ** 6 / phi ** 2)
    Ne = (Nw*torch.fmin(1/g, phi/Bth**2)/eta_sp + Nw**3*torch.fmin(1/g, phi/Bth**2)/eta_sp) ** 0.5

    popt_arr = torch.tensor(np.zeros(shape=(batch_size,1)))

    for i in range(0, batch_size):

        if torch.unique(Bg[i][Bg[i] != 0])[0] <= torch.unique(Bth[i][Bth[i]!=0])[0] ** (4-6*0.588):

            x = torch.where(phi[i] < Bth[i]**4, g[i], Bth[i]**6/phi[i]**2)
            x = x[(~Ne[i].isnan()) & (~Ne[i].isinf())]
            y = Ne[i][(~Ne[i].isnan()) & (~Ne[i].isinf())]

            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            popt, pcov = curve_fit(pe_fit_function, x, y)
            val = popt[0]
            popt_arr[i] = val

        else:

            x = torch.where(phi[i]<Bth[i]**3*(Bth[i]/Bg[i])**(1/(2*0.588-1)),g[i]*Bth[i]**3*(Bth[i]/Bg[i])**(1/(2*0.588-1)),torch.where(phi[i]<Bth[i]**6,Bth[i]**2*phi[i]**(-4/3),torch.where(phi[i] < Bth[i]**4, g[i], Bth[i]**6/phi[i]**2)))
            x = x[(~Ne[i].isnan()) & (~Ne[i].isinf())]
            y = Ne[i][(~Ne[i].isnan()) & (~Ne[i].isinf())]
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            popt, pcov = curve_fit(pe_fit_function, x, y)
            val = popt[0]
            popt_arr[i] = val

    #Ne = torch.where(eta_sp < 1, 0, Ne)

    #lam_g_g = torch.fmin(torch.fmin(torch.fmin(g*phi**(2/3)/Bth**4, Bth**2*phi**(4/3)),g), Bth**6/phi**2)
    #lam_g_g = torch.where(eta_sp < 1, 0, lam_g_g)

    #Bg = torch.where(g < 1, 0, Bg)
    #Bth = torch.where(g < 1, 0, Bth)
    #g = torch.where(g < 1, 0, g)
    #Ne = torch.where(g < 1, 0, Ne)
    #lam_g_g = torch.where(g < 1, 0, lam_g_g)


        #ne_slice1 = Ne[i][eta_sp[i] > 1]
        #lam_g_g_slice1 = Ne[i][eta_sp
        #Ne_slice = Ne[i][eta_sp[i] != 0]
        #lam_g_g_slice = lam_g_g[i][eta_sp[i] != 0]
        #Ne_filter = Ne_slice[(~Ne_slice.isnan()) & (~lam_g_g_slice.isnan()) & (~Ne_slice.isinf()) & (~lam_g_g_slice.isinf())]
        #lam_g_g_filter = lam_g_g[i][eta_sp[i] != 0][(~Ne_slice.isnan()) & (~lam_g_g_slice.isnan()) & (~Ne_slice.isinf()) & (~lam_g_g_slice.isinf())]

        #model = linearRegression()
        #criterion = torch.nn.MSELoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

    return popt_arr
