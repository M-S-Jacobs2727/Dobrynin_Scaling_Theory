import matplotlib.pyplot as plt
import torch
import numpy as np
import scaling_torch_lib as mike
import pandas as pd

PHI_MIN = 3e-5
PHI_MAX = 2e-1
NW_MIN = 25
NW_MAX = 2.3e5
NUM_BINS = 96

def plot_data(eval_y, eval_pred):

    bg_true = eval_y[:,0].cpu().numpy()
    bth_true = eval_y[:,1].cpu().numpy()
    pe_true = eval_y[:,2].cpu().numpy()

    bg_pred = eval_pred[:,0].cpu().numpy()
    bth_pred = eval_pred[:,1].cpu().numpy()
    pe_pred = eval_pred[:,2].cpu().numpy()

    max_bg = np.amax(np.maximum(bg_true,bg_pred)) 
    max_bth = np.amax(np.maximum(bth_true,bth_pred)) 
    max_pe = np.amax(np.maximum(pe_true,pe_pred)) 

    max_bg_figure = max_bg * 1.1
    max_bth_figure = max_bth * 1.1
    max_pe_figure = max_pe * 1.1

    bg_line = (0, max_bg)
    bth_line = (0, max_bth)
    pe_line = (0, max_pe)

    fig, (ax_bg, ax_bth, ax_pe) = plt.subplots(1,3,figsize=(12,3))
    ax_bg.scatter(bg_true, bg_pred, c='b', s=8)
    ax_bth.scatter(bth_true, bth_pred,c='b', s=8)
    ax_pe.scatter(pe_true, pe_pred,c='b', s=8)

    ax_bg.plot([0,max_bg_figure], [0,max_bg_figure], 'k-')
    ax_bth.plot([0,max_bth_figure], [0,max_bth_figure], 'k-')
    ax_pe.plot([0,max_pe_figure], [0,max_pe_figure], 'k-')

    plots = [ax_bg, ax_bth, ax_pe]
    for plot in plots:
        plot.set_xlabel('true value')
        plot.set_ylabel('predicted value')
        plot.set_aspect('equal')

    ax_bg.set_xlim(0, max_bg_figure)
    ax_bg.set_ylim(0, max_bg_figure)
    ax_bg.set_title('Bg')

    ax_bth.set_xlim(0, max_bth_figure)
    ax_bth.set_ylim(0, max_bth_figure)
    ax_bth.set_title('Bth')

    ax_pe.set_xlim(0, max_pe_figure)
    ax_pe.set_ylim(0, max_pe_figure)
    ax_pe.set_title('Pe')

    plt.show()

    plt.close()

def record_fits(df, pred_unnorm_data, eta_sp_grid, eta_131_norm, eta_2_norm, batch_size, device):

    phi_grid = torch.tensor(np.geomspace(
        PHI_MIN,
        PHI_MAX,
        NUM_BINS,
        endpoint=True
    ), dtype=torch.float, device=device)

    Nw_grid = torch.tensor(np.geomspace(
        NW_MIN,
        NW_MAX,
        NUM_BINS,
        endpoint=True
    ), dtype=torch.float, device=device)

    phi_grid, Nw_grid = torch.meshgrid(phi_grid, Nw_grid, indexing='xy')
    phi_grid = torch.tile(phi_grid, (batch_size, 1, 1))
    Nw_grid = torch.tile(Nw_grid, (batch_size, 1, 1))


    features = ['Nw', 'phi', 'eta_sp', 'eta_sp_131', 'eta_sp_2']
    params = ['Bg', 'Bth', 'Pe']
    df2 = df.copy()
    df2 = df2[(df2['Bg'] > 0) & (df2['Bth'] > 0)]

    counter = 0
    for i in np.unique(df2['group']):

        polymer = np.unique(df2[df2["group"] == i]["Polymer"])[0]
        solvent = np.unique(df2[df2["group"] == i]["Solvent"])[0]

        # get experimental data and normalizations
        data_slice = df2[df2['group']==i][features]
        data_slice_params = df2[df2['group']==i][params]
        phi_exp = torch.tensor(data_slice['phi'].values).to(device)
        eta_exp = torch.tensor(data_slice['eta_sp'].values).to(device)
        eta_131_exp = data_slice['eta_sp_131']
        eta_2_exp = data_slice['eta_sp_2']
        Bg = torch.tensor(np.unique(data_slice_params["Bg"])).to(device)
        Bth = torch.tensor(np.unique(data_slice_params["Bth"])).to(device)
        Pe_exp = torch.tensor(np.unique(data_slice_params['Pe'])).to(device)
        Nw = torch.tensor(np.unique(data_slice['Nw'])).to(device)
        Nw_exp = torch.tensor(data_slice['Nw'].values).to(device)

        phi_star_star_exp = Bth**4
        lam_g_exp = torch.fmin(
                torch.tensor([1], device=device), torch.fmin(
                    (Bth / Bg) ** (2 / (6 * 0.588 - 3)) / Bth**2,
                    Bth ** 4 * phi_exp**(2 / 3)
                    )
                )
        g_exp = torch.fmin(Bg**(3/0.764) / phi_exp**(1/0.764), Bth / phi_exp**2)
        lam_g_g_exp = lam_g_exp * g_exp

        lam_exp = 1 / lam_g_exp * torch.where(phi_exp < phi_star_star_exp, 1, (phi_exp / phi_star_star_exp) ** 0.5)

        Nw_lam_g_g_exp = Nw_exp / lam_g_g_exp
        lam_eta_sp_exp = eta_exp * lam_exp

        Ne_exp = Pe_exp**2 * g_exp * torch.fmin(
                torch.tensor([1], device=device), torch.fmin(
                (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
                Bth**4 * phi_exp**(2/3)
                        )
                )
        x_exp = lam_g_g_exp.detach().cpu().numpy()
        y_exp = Ne_exp.detach().cpu().numpy()


        # get grid data and normalizations
        phi_plot = phi_grid[counter][eta_sp_grid[counter] != 0]
        nw_plot = Nw_grid[counter][eta_sp_grid[counter] != 0]
        eta_sp_plot = eta_sp_grid[counter][eta_sp_grid[counter] != 0]
        eta_sp_131_plot = eta_131_norm[counter][eta_sp_grid[counter] != 0]
        eta_sp_2_plot = eta_2_norm[counter][eta_sp_grid[counter] != 0]

        num_nw = torch.unique(Nw_exp).size()[0]

        lam_g_plot = torch.fmin(
                torch.tensor([1], device=device), torch.fmin(
                    (Bth / Bg) ** (2 / (6 * 0.588 - 3)) / Bth**2,
                    Bth ** 4 * phi_plot**(2/3)
                    )
                )

        g_plot = torch.fmin(Bg**(3/0.764) / phi_plot**(1/0.764), Bth / phi_plot**2)
        lam_g_g_plot = lam_g_plot * g_plot
        lam_plot = 1 / lam_g_plot * torch.where(phi_plot < phi_star_star_exp, 1, (phi_plot / phi_star_star_exp) ** 0.5)

        Ne_plot = Pe_exp**2 * g_plot * torch.fmin(
                torch.tensor([1], device=device), torch.fmin(
                (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
                Bth**4 * phi_plot**(2/3)
                        )
                )

        Nw_lam_g_g_plot = nw_plot / lam_g_g_plot
        lam_eta_sp_plot = eta_sp_plot * lam_plot

        x_plot = lam_g_g_plot.detach().cpu().numpy()
        y_plot = Ne_plot.detach().cpu().numpy()

        # get predicted data and normalizations

        #phi = torch.tensor(np.geomspace(PHI_MIN, PHI_MAX, NUM_BINS)).to(device)
        #nw_dummy = torch.tensor(np.geomspace(NW_MIN, NW_MAX, NUM_BINS)).to(device)

        num_nw = torch.unique(Nw_exp).size()[0]
        if num_nw == 1:
            nw = torch.unique(Nw_exp)[0]
            phi = torch.tensor(np.geomspace(PHI_MIN, PHI_MAX, NUM_BINS)).to(device)
        else:
            nw = Nw_grid[counter][eta_sp_grid[counter] != 0]
            phi = phi_grid[counter][eta_sp_grid[counter] != 0]

        Bg_pred = pred_unnorm_data[counter, 0]
        Bth_pred = pred_unnorm_data[counter, 1]
        Pe = pred_unnorm_data[counter, 2]
        g = torch.fmin(Bg_pred**(3/0.764) / phi**(1/0.764), Bth_pred**6 / phi**2)
        Ne = Pe**2 * g * torch.fmin(
                torch.tensor([1], device=device), torch.fmin(
                (Bth_pred / Bg_pred)**(2/(6*0.588 - 3)) / Bth_pred**2,
                Bth_pred**4 * phi**(2/3)
                        )
                )

        eta_sp_pred = nw * (1 + (nw / Ne)**2) * torch.fmin(1/g,phi / Bth_pred**2)

        eta_sp_131_pred = eta_sp_pred / nw / phi**(1/(3*0.588-1))
        eta_sp_2_pred = eta_sp_pred / nw / phi**2

        lam_g = torch.fmin(
            torch.tensor([1], device=device), torch.fmin(
                (Bth_pred / Bg_pred)**(2/(6*0.588 - 3)) / Bth_pred**2,
                Bth_pred**4 * phi**(2/3)
            )
        )

        lam_g_g = lam_g * g
        phi_star_star_pred = Bth_pred**4
        lam = 1 / lam_g * torch.where(phi < phi_star_star_pred, 1, (phi / phi_star_star_pred) ** 0.5)

        Nw_lam_g_g_pred = nw / lam_g_g
        lam_eta_sp_pred = eta_sp_pred * lam

        x_pred = lam_g_g.detach().cpu().numpy()
        y_pred = Ne.detach().cpu().numpy()

        # move to cpu
        phi_exp = phi_exp.detach().cpu().numpy()
        eta_exp = eta_exp.detach().cpu().numpy()
        phi_plot = phi_plot.detach().cpu().numpy()
        eta_sp_plot = eta_sp_plot.detach().cpu().numpy()
        eta_sp_131_plot = eta_sp_131_plot.detach().cpu().numpy()
        eta_sp_2_plot = eta_sp_2_plot.detach().cpu().numpy()
        phi = phi.detach().cpu().numpy()
        eta_sp_pred = eta_sp_pred.detach().cpu().numpy()
        eta_sp_131_pred = eta_sp_131_pred.detach().cpu().numpy()
        eta_sp_2_pred = eta_sp_2_pred.detach().cpu().numpy()
        Pe = Pe.detach().cpu().numpy()
        Nw_lam_g_g_pred = Nw_lam_g_g_pred.detach().cpu().numpy()
        lam_eta_sp_pred = lam_eta_sp_pred.detach().cpu().numpy()
        Nw_lam_g_g_plot = Nw_lam_g_g_plot.detach().cpu().numpy()
        lam_eta_sp_plot = lam_eta_sp_plot.detach().cpu().numpy()
        Nw_lam_g_g_exp = Nw_lam_g_g_exp.detach().cpu().numpy()
        lam_eta_sp_exp = lam_eta_sp_exp.detach().cpu().numpy()
        folder = 'fitting_results/'
        # plot viscosity plot
        plt.scatter(phi_plot, eta_sp_plot, color='blue', label='grid data')
        plt.scatter(phi_exp, eta_exp, color='red', label='exp data')
        plt.plot(phi, eta_sp_pred, color='black', label='predicted data')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{folder}{polymer}_{solvent}_{counter}_{Pe:.3f}.tiff')
        plt.clf()

        # plot 131 normalization plot
        plt.scatter(phi_plot, eta_sp_131_plot, color='blue', label='grid data')
        plt.scatter(phi_exp, eta_131_exp, color='red', label='exp data')
        plt.plot(phi, eta_sp_131_pred, color='black', label='predicted data')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{folder}{polymer}_{solvent}_{counter}_{Pe:.3f}_131.tiff')
        plt.clf()

        # plot 2 normalization plot 
        plt.scatter(phi_plot, eta_sp_2_plot, color='blue', label='grid data')
        plt.scatter(phi_exp, eta_2_exp, color='red', label='exp data')
        plt.plot(phi, eta_sp_2_pred, color='black', label='predicted data')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{folder}{polymer}_{solvent}_{counter}_{Pe:.3f}_2.tiff')
        plt.clf()

        # plot lam_eta_sp vs Nw/lam_g_g
        plt.scatter(Nw_lam_g_g_plot, lam_eta_sp_plot, color='blue', label='grid data')
        plt.scatter(Nw_lam_g_g_exp, lam_eta_sp_exp, color='red', label='exp data')
        plt.plot(Nw_lam_g_g_pred, lam_eta_sp_pred, color='black', label='predicted data')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{folder}{polymer}_{solvent}_{counter}_{Pe:.3f}_uni.tiff')
        plt.clf() 

        # plot fit of Ne vs lam_g_g
        plt.scatter(x_plot, y_plot, color='blue', label='grid data')
        plt.scatter(x_exp, y_exp, color='red', label='exp data')
        plt.plot(x_pred, y_pred, color='black', label='predicted data')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{folder}{polymer}_{solvent}_{counter}_{Pe:.3f}_fit.tiff')
        plt.clf() 

        counter = counter + 1
