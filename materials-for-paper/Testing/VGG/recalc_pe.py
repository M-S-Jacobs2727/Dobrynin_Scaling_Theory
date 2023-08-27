import collections
import numpy as np
import torch
import random
import torch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(3e-5, 2e-2)
NW = Param(100, 1e5)
ETA_SP = Param(1, 1e6)
BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(3.2, 13.5)

ETA_SP_131 = Param(ETA_SP.min/NW.max/PHI.max**(1/(3*0.588-1)),ETA_SP.max/NW.min/PHI.min**(1/(3*0.588-1)))
ETA_SP_2 = Param(ETA_SP.min/NW.max/PHI.max**2,ETA_SP.max/NW.min/PHI.min**2)

def pe_fit_master_curve(x, Pe):

    return x*(1+(x/Pe**2)**2)

def pe_fit_master_curve_charged(x, Pe):

    return x*(1+(x/Pe**2))**2

def get_pe(table_vals, df, device):

    folder = "pe_fit_results_Human_Guess"

    df2 = df.copy()
    # remember to cut out points that lie outside our (Nw, phi, eta_sp) range!
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]

    popt_arr = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))
    pe_true = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))

    counter = 0

    for i in np.unique(table_vals['Group']):

        df_slice = df2[df2['group']==i]
        if df_slice['Bg'].iloc[0] > 0:
            df_slice['Bg'] = table_vals['Pred Bg'][table_vals['Group']==i].iloc[0]
            df2['Bg'][df2['group']==i] = df_slice['Bg'].iloc[0]

        if df_slice['Bth'].iloc[0] > 0:
            df_slice['Bth'] = table_vals['Pred Bth'][table_vals['Group']==i].iloc[0]
            df2['Bth'][df2['group']==i] = df_slice['Bth'].iloc[0]

    for i in np.unique(table_vals['Subgroup']):

        table_eta_sp_slice = df2[df2['subgroup']==i]
        pe_true[counter] = df2['Pe'][df2['subgroup']==i].iloc[0]

        phi = torch.tensor(table_eta_sp_slice['phi'].values).to(device)
        Nw = torch.tensor(table_eta_sp_slice['Nw'].values).to(device)
        eta_sp = torch.tensor(table_eta_sp_slice['eta_sp'].values).to(device)
        Bg = torch.tensor(table_eta_sp_slice['Bg'].values).to(device)
        Bth = torch.tensor(table_eta_sp_slice['Bth'].values).to(device)

        Bg[Bg==0] = torch.nan
        Bth[Bth==0] = torch.nan

        g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth ** 6 / phi ** 2)

        mask = g > 1

        mask_bg = Bg > 0
        phi_star = torch.clone(eta_sp)
        phi_star[mask_bg] = (Bg[mask_bg]**3*Nw[mask_bg]**(1-3*0.588)).to(device)
        phi_star[~mask_bg] = (Bth[~mask_bg]**3*Nw[~mask_bg]**(-0.5)).to(device)
        phi_th = (Bth**3*(Bth/Bg)**(1/(2*0.588-1))).to(device)
        Bth6 = (Bth**6).to(device)
        phi_star_star = (Bth**4).to(device)
        param = ((Bth/Bg)**(1/(2*0.588-1))/Bth**3).to(device)

        lam_g_KN = torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))
        lam_g_RC = torch.where(phi<phi_th, phi_th**(2/3)*Bth**-4, torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))))
        lam_g_Bth = torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2)))

        lam_g_g = g * torch.where(param < 1, lam_g_RC, lam_g_KN)
        if table_eta_sp_slice['Bg'].iloc[0] == 0:
            lam_g_g = g * lam_g_Bth
        if table_eta_sp_slice['Bth'].iloc[0] == 0:
            lam_g_g = g
     
        lam = 1 / torch.where(param < 1, lam_g_RC, lam_g_KN) * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        if table_eta_sp_slice['Bg'].iloc[0] == 0:
            lam = 1 / lam_g_Bth * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))

        if table_eta_sp_slice['Bth'].iloc[0] == 0:
            lam[:] = 1

        x = Nw[mask] / lam_g_g[mask]
        y = eta_sp[mask] * lam[mask]

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        popt, pcov = curve_fit(pe_fit_master_curve, x, y, sigma=1/y, method='lm')

        table_vals['Pred Pe'][table_vals['Subgroup']==i] = np.around(popt[0], 3)

        x_fit = np.geomspace(0.5, 1e5, 100000)
        pe_val = popt[0]
        y_fit = x_fit*(1+(x_fit/pe_val**2)**2)

        plt.scatter(x, y)
        plt.plot(x_fit, y_fit, color='black')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(0.5,1e4)
        plt.ylim(0.5,1e7)
        plt.xlabel('Nw/lam_g_g')
        plt.ylabel('eta_sp*lam')
        title=table_eta_sp_slice['Polymer'].iloc[0],table_eta_sp_slice['Solvent'].iloc[0]
        plt.title(title)
        plt.savefig(f'{folder}/pe_fit_{counter}_curve.png')
        plt.clf()

        popt_arr[counter] = np.around(popt[0], 3)

        counter += 1

    pe_true = pe_true.detach().cpu().numpy()
    popt_arr = popt_arr.detach().cpu().numpy()

    pe_error = np.average(np.abs(pe_true-popt_arr)/pe_true*100)

    return table_vals, pe_error

def get_pe_combo(table_vals, df, device):

    folder = "pe_fit_results_combo"

    table_vals['Pred Pe Combo'] = 0

    df2 = df.copy()
    # remember to cut out points that lie outside our (Nw, phi, eta_sp) range!
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]

    popt_arr = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))
    pe_true = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))

    counter = 0

    for i in np.unique(table_vals['Group']):

        df_slice = df2[df2['group']==i]

        df_slice['Bg'] = table_vals['Pred Bg'][table_vals['Group']==i].iloc[0]
        df2['Bg'][df2['group']==i] = df_slice['Bg'].iloc[0]

        df_slice['Bth'] = table_vals['Pred Bth'][table_vals['Group']==i].iloc[0]
        df2['Bth'][df2['group']==i] = df_slice['Bth'].iloc[0]

    for i in np.unique(table_vals['Subgroup']):

        table_eta_sp_slice = df2[df2['subgroup']==i]
        pe_true[counter] = df2['Pe'][df2['subgroup']==i].iloc[0]

        phi = torch.tensor(table_eta_sp_slice['phi'].values).to(device)
        Nw = torch.tensor(table_eta_sp_slice['Nw'].values).to(device)
        eta_sp = torch.tensor(table_eta_sp_slice['eta_sp'].values).to(device)
        Bg = torch.tensor(table_eta_sp_slice['Bg'].values).to(device)
        Bth = torch.tensor(table_eta_sp_slice['Bth'].values).to(device)

        #print(Bth)

        #Bg[Bg==0] = torch.nan
        #Bth[Bth==0] = torch.nan

        g = torch.fmin(Bg ** (3 / 0.764) / phi ** (1 / 0.764), Bth ** 6 / phi ** 2)

        mask = g > 1

        mask_bg = Bg > 0
        phi_star = torch.clone(eta_sp)
        phi_star[mask_bg] = (Bg[mask_bg]**3*Nw[mask_bg]**(1-3*0.588)).to(device)
        phi_star[~mask_bg] = (Bth[~mask_bg]**3*Nw[~mask_bg]**(-0.5)).to(device)
        phi_th = (Bth**3*(Bth/Bg)**(1/(2*0.588-1))).to(device)
        Bth6 = (Bth**6).to(device)
        phi_star_star = (Bth**4).to(device)
        param = ((Bth/Bg)**(1/(2*0.588-1))/Bth**3).to(device)

        lam_g_KN = torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))
        lam_g_RC = torch.where(phi<phi_th, phi_th**(2/3)*Bth**-4, torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))))
        lam_g_Bth = torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2)))

        lam_g_g = g * torch.where(param < 1, lam_g_RC, lam_g_KN)
        #if table_eta_sp_slice['Bg'].iloc[0] == 0:
        #    lam_g_g = g * lam_g_Bth
        #if table_eta_sp_slice['Bth'].iloc[0] == 0:
        #    lam_g_g = g
     
        lam = 1 / torch.where(param < 1, lam_g_RC, lam_g_KN) * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))
        #if table_eta_sp_slice['Bg'].iloc[0] == 0:
        #    lam = 1 / lam_g_Bth * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))

        #if table_eta_sp_slice['Bth'].iloc[0] == 0:
        #    lam[:] = 1

        x = Nw[mask] / lam_g_g[mask]
        y = eta_sp[mask] * lam[mask]

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        popt, pcov = curve_fit(pe_fit_master_curve, x, y, sigma=1/y, method='lm')

        table_vals['Pred Pe Combo'][table_vals['Subgroup']==i] = np.around(popt[0], 3)

        x_fit = np.geomspace(0.5, 1e5, 100000)
        pe_val = popt[0]
        y_fit = x_fit*(1+(x_fit/pe_val**2)**2)

        plt.scatter(x, y)
        plt.plot(x_fit, y_fit, color='black')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(0.5,1e4)
        plt.ylim(0.5,1e7)
        plt.xlabel('Nw/lam_g_g')
        plt.ylabel('eta_sp*lam')
        plt.savefig(f'{folder}/pe_fit_{counter}_curve.png')
        plt.clf()

        popt_arr[counter] = np.around(popt[0], 3)

        counter += 1

    #pe_true = pe_true.detach().cpu().numpy()
    #popt_arr = popt_arr.detach().cpu().numpy()

    #pe_error = np.average(np.abs(pe_true-popt_arr)/pe_true*100)

    return table_vals

def get_pe_g(table_vals, df, device):

    folder = "pe_fit_results_g"

    table_vals['Pred Pe g'] = 0

    df2 = df.copy()
    # remember to cut out points that lie outside our (Nw, phi, eta_sp) range!
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]

    popt_arr = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))
    #pe_true = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))

    counter = 0

    for i in np.unique(table_vals['Group']):

        df_slice = df2[df2['group']==i]

        df_slice['Bg'] = table_vals['Pred Bg'][table_vals['Group']==i].iloc[0]
        df2['Bg'][df2['group']==i] = df_slice['Bg'].iloc[0]

        #df_slice['Bth'] = table_vals['Pred Bth'][table_vals['Group']==i].iloc[0]
        #df2['Bth'][df2['group']==i] = df_slice['Bth'].iloc[0]

    for i in np.unique(table_vals['Subgroup']):

        table_eta_sp_slice = df2[df2['subgroup']==i]
        #pe_true[counter] = df2['Pe'][df2['subgroup']==i].iloc[0]

        phi = torch.tensor(table_eta_sp_slice['phi'].values).to(device)
        Nw = torch.tensor(table_eta_sp_slice['Nw'].values).to(device)
        eta_sp = torch.tensor(table_eta_sp_slice['eta_sp'].values).to(device)
        Bg = torch.tensor(table_eta_sp_slice['Bg'].values).to(device)
        #Bth = torch.tensor(table_eta_sp_slice['Bth'].values).to(device)

        g = Bg ** (3 / 0.764) / phi ** (1 / 0.764)

        mask = g > 1

        phi_star = torch.clone(eta_sp)
        phi_star = (Bg**3*Nw**(1-3*0.588)).to(device)

        x = Nw[mask] / g[mask]
        y = eta_sp[mask]

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        popt, pcov = curve_fit(pe_fit_master_curve, x, y, sigma=1/y, method='lm')

        table_vals['Pred Pe g'][table_vals['Subgroup']==i] = np.around(popt[0], 3)

        x_fit = np.geomspace(0.5, 1e5, 100000)
        pe_val = popt[0]
        y_fit = x_fit*(1+(x_fit/pe_val**2)**2)

        plt.scatter(x, y)
        plt.plot(x_fit, y_fit, color='black')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(0.5,1e4)
        plt.ylim(0.5,1e7)
        plt.xlabel('Nw/g')
        plt.ylabel('eta_sp')
        plt.savefig(f'{folder}/pe_fit_{counter}_curve.png')
        plt.clf()

        popt_arr[counter] = np.around(popt[0], 3)

        counter += 1

    #pe_true = pe_true.detach().cpu().numpy()
    popt_arr = popt_arr.detach().cpu().numpy()

    return table_vals

def get_pe_th(table_vals, df, device):

    folder = "pe_fit_results_th"

    table_vals['Pred Pe th'] = 0

    df2 = df.copy()
    # remember to cut out points that lie outside our (Nw, phi, eta_sp) range!
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]

    popt_arr = torch.tensor(np.zeros(shape=(len(np.unique(table_vals['Subgroup'])),1)))

    counter = 0

    for i in np.unique(table_vals['Group']):

        df_slice = df2[df2['group']==i]

        df_slice['Bth'] = table_vals['Pred Bth'][table_vals['Group']==i].iloc[0]
        df2['Bth'][df2['group']==i] = df_slice['Bth'].iloc[0]

    for i in np.unique(table_vals['Subgroup']):

        table_eta_sp_slice = df2[df2['subgroup']==i]
        #pe_true[counter] = df2['Pe'][df2['subgroup']==i].iloc[0]

        phi = torch.tensor(table_eta_sp_slice['phi'].values).to(device)
        Nw = torch.tensor(table_eta_sp_slice['Nw'].values).to(device)
        eta_sp = torch.tensor(table_eta_sp_slice['eta_sp'].values).to(device)
        #Bg = torch.tensor(table_eta_sp_slice['Bg'].values).to(device)
        Bth = torch.tensor(table_eta_sp_slice['Bth'].values).to(device)

        #Bth[Bth==0] = torch.nan

        g = Bth ** 6 / phi ** 2

        mask = g > 1

        phi_star = torch.clone(eta_sp)
        phi_star = (Bth**3*Nw**(-0.5)).to(device)
        #phi_th = (Bth**3*(Bth/Bg)**(1/(2*0.588-1))).to(device)
        Bth6 = (Bth**6).to(device)
        phi_star_star = (Bth**4).to(device)
        #param = ((Bth/Bg)**(1/(2*0.588-1))/Bth**3).to(device)

        #lam_g_KN = torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))
        #lam_g_RC = torch.where(phi<phi_th, phi_th**(2/3)*Bth**-4, torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2))))
        lam_g_Bth = torch.where(phi<Bth6, phi**(2/3)*Bth**-4, torch.where(phi<phi_star_star, 1, (phi/phi_star_star)**(-3/2)))

        lam_g_g = g * lam_g_Bth

        lam = 1 / lam_g_Bth * torch.where(phi < phi_star_star, 1, (phi/phi_star_star)**(-1/2))

        x = Nw[mask] / lam_g_g[mask]
        y = eta_sp[mask] * lam[mask]

        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        popt, pcov = curve_fit(pe_fit_master_curve, x, y, sigma=1/y, method='lm')

        table_vals['Pred Pe th'][table_vals['Subgroup']==i] = np.around(popt[0], 3)

        x_fit = np.geomspace(0.5, 1e5, 100000)
        pe_val = popt[0]
        y_fit = x_fit*(1+(x_fit/pe_val**2)**2)

        plt.scatter(x, y)
        plt.plot(x_fit, y_fit, color='black')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlim(0.5,1e4)
        plt.ylim(0.5,1e7)
        plt.xlabel('Nw/lam_g_g')
        plt.ylabel('eta_sp*lam')
        plt.savefig(f'{folder}/pe_fit_{counter}_curve.png')
        plt.clf()

        popt_arr[counter] = np.around(popt[0], 3)

        counter += 1

    #pe_true = pe_true.detach().cpu().numpy()
    popt_arr = popt_arr.detach().cpu().numpy()

    #pe_error = np.average(np.abs(pe_true-popt_arr)/pe_true*100)

    return table_vals
