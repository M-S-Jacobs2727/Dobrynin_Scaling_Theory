import numpy as np
import pandas as pd
import torch
import collections
import json
from googlenet_serial import InceptionV1
import scipy.interpolate
import loss_functions
import plot_eval
import scaling_torch_lib as mike
import warnings
from pandas.core.common import SettingWithCopyWarning
import recalc_pe as ryan
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
Param = collections.namedtuple('Param', ('min', 'max'))

#PHI = Param(1e-6, 1e-2)
#PHI = Param(3e-5, 2e-1)
PHI = Param(3e-5, 2e-1)
NW = Param(25, 2.3e5)
ETA_SP = Param(1, 3.1e6)

BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(2.5, 13.5)

ETA_SP_131 = Param(0.1, 150000)
ETA_SP_2 = Param(1, 4000000)

NUM_BIN = 96

def get_grid_single_nw_data(nw_norm_val, nw_plot_bins, eta_new):

    eta_grid = np.zeros([len(nw_plot_bins), len(nw_plot_bins)])
    idx = np.abs(nw_plot_bins-nw_norm_val).argmin()
    eta_grid[:,idx] = eta_new
    return eta_grid

def get_nw_idx_single(nw_norm, nw_plot_bins):

    nw_norm_uniq = nw_norm.unique()  
    idx = np.abs(nw_plot_bins - nw_norm_uniq).argmin()

    return idx

def generate_surface_bins():
    """
    Read json file "surface_bins.json" to output bins to create (nw, phi, eta_sp) space

        Arguments:
            None

        Returns:
            Parameters: (dict) Dictionary of tuples. Number of bins [0], minimum bin [1], and maximum bin [2] for each coordinate of Nw, phi, and eta_sp
    """

    with open('surface_bins.json') as f:
        bin_data = json.load(f)

    bins = {}
    bins['Nw'] = Bin(bin_data['Nw_min_bin'], bin_data['Nw_max_bin'], bin_data['Nw_num_bins'])
    bins['phi'] = Bin(bin_data['phi_min_bin'], bin_data['phi_max_bin'], bin_data['phi_num_bins'])
    bins['eta_sp'] = Bin(bin_data['eta_sp_min_bin'], bin_data['eta_sp_max_bin'], bin_data['eta_sp_num_bins'])

    return bins

def normalize_params(y):

    y_new = torch.clone(y)
    y_new[:,0] = (y_new[:,0]-BG.min) / (BG.max - BG.min)
    y_new[:,1] = (y_new[:,1]-BTH.min) / (BTH.max - BTH.min)
    y_new[:,2] = (y_new[:,2]-PE.min) / (PE.max - PE.min) 
    return y_new

def normalize_bg_bth(y):

    y[:,0] = (y[:,0]-BG.min) / (BG.max - BG.min)
    y[:,1] = (y[:,1]-BTH.min) / (BTH.max - BTH.min)
    return y 

def normalize_visc(eta_sp):
    """
    take log and normalize. Note here this function does not add noise or cap values
    """
    
    return (torch.log(eta_sp) - torch.log(ETA_SP.min)) / \
            (torch.log(ETA_SP.max) - torch.log(ETA_SP.min))

def normalize_data(nw, phi, eta, eta_131, eta_2):

    nw = (np.log(nw) - np.log(NW.min)) / (np.log(NW.max) - np.log(NW.min))
    phi = (np.log(phi) - np.log(PHI.min)) / (np.log(PHI.max) - np.log(PHI.min))
    eta = (np.log(eta) - np.log(ETA_SP.min)) / (np.log(ETA_SP.max) - np.log(ETA_SP.min))
    eta_131 = (np.log(eta_131) - np.log(ETA_SP_131.min)) / (np.log(ETA_SP_131.max) - np.log(ETA_SP_131.min))
    eta_2 = (np.log(eta_2) - np.log(ETA_SP_2.min)) / (np.log(ETA_SP_2.max) - np.log(ETA_SP_2.min))
    return nw, phi, eta, eta_131, eta_2

def unnorm_pe(eta_sp_norm):

    eta_sp_unnorm = ETA_SP.min*torch.exp(eta_sp_norm*torch.log(torch.tensor(ETA_SP.max)/torch.tensor(ETA_SP.min)))
    return eta_sp_unnorm

def load_exp_data(df, device, resolution, batch_size):

    features = ['Nw', 'phi', 'eta_sp', 'eta131', 'eta2']
    params = ['Bg', 'Bth', 'Pe']

    df2 = df.copy()
  
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]
    df2["eta131"] = df2["eta_sp"] / df2["Nw"] / df2["phi"] ** (1 / (3 * 0.588 - 1))
    df2["eta2"] = df2["eta_sp"] / df2["Nw"] / df2["phi"] ** 2

    df2 = df2[(df2['Bg'] > 0) & (df2['Bth'] > 0)]

    num_points = len(np.unique(df2['group']))
    init = np.zeros(num_points)
    # Create data for y true vals
    labels = torch.zeros(len(np.unique(df2['group'])), 3).to(device)
    X = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN).to(device)
    X_131 = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN).to(device)
    X_2 = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN).to(device)

    # new code # 
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

    counter = 0
    df_create = pd.DataFrame( 
            { 
                "Polymer": pd.Series(data=init,dtype="str"), 
                "Solvent": pd.Series(data=init,dtype="str"), 
                "True Bg": pd.Series(data=init,dtype="float"), 
                "Pred Bg": pd.Series(data=init,dtype="float"), 
                "True Bth": pd.Series(data=init,dtype="float"), 
                "Pred Bth": pd.Series(data=init,dtype="float"), 
                "True Pe": pd.Series(data=init,dtype="float"), 
                "Pred Pe": pd.Series(data=init,dtype="float"), 
                "Num Nw": pd.Series(data=init,dtype="int"), 
             } 
        )  

    for i in np.unique(df2['group']):

        polymer = np.unique(df2[df2["group"] == i]["Polymer"])[0]
        solvent = np.unique(df2[df2["group"] == i]["Solvent"])[0]

        df_create["Polymer"][counter] = polymer
        df_create["Solvent"][counter] = solvent

        data_slice = df2[df2['group']==i][features]
        data_slice_params = df2[df2['group']==i][params]
        df_create["Num Nw"][counter] = len(np.unique(data_slice['Nw']))

        nw_exp = data_slice['Nw']
        phi_exp = data_slice['phi']
        eta_exp = data_slice['eta_sp']
        eta131_exp = data_slice['eta131']
        eta2_exp = data_slice['eta2']

        # normalize
        #eta_131 = eta/nw/phi**(1/(3*0.588-1))
        #eta_2 = eta/nw/phi**2

        nw_norm, phi_norm, eta_norm, eta_131_norm, eta_2_norm = normalize_data(nw_exp, phi_exp, eta_exp, eta131_exp, eta2_exp)

        # Retrieve Nw indices

        nw_plot_bins = np.linspace(0,1,NUM_BIN)
        phi_plot_bins = np.linspace(0,1,NUM_BIN)

        phi_plot, nw_plot = np.meshgrid(phi_plot_bins, nw_plot_bins, indexing='xy')

        if len(np.unique(data_slice['Nw'])) > 1:

            eta_grid_data = scipy.interpolate.griddata(np.array([phi_norm.ravel(),nw_norm.ravel()]).T,
                    eta_norm.ravel(),np.array([phi_plot.ravel(),nw_plot.ravel()]).T,
                    method='linear', fill_value=0)
            data_save = eta_grid_data.reshape(NUM_BIN,NUM_BIN)

            eta131_grid_data = scipy.interpolate.griddata(np.array([phi_norm.ravel(),nw_norm.ravel()]).T,
                    eta_131_norm.ravel(),np.array([phi_plot.ravel(),nw_plot.ravel()]).T,
                    method='linear', fill_value=0)
            data131_save = eta131_grid_data.reshape(NUM_BIN,NUM_BIN)

            eta2_grid_data = scipy.interpolate.griddata(np.array([phi_norm.ravel(),nw_norm.ravel()]).T,
                    eta_2_norm.ravel(),np.array([phi_plot.ravel(),nw_plot.ravel()]).T,
                    method='linear', fill_value=0)
            data2_save = eta2_grid_data.reshape(NUM_BIN,NUM_BIN)

            for j in range(0,NUM_BIN):
                X[counter, j] = torch.from_numpy(data_save[j])
                X_131[counter, j] = torch.from_numpy(data131_save[j])
                X_2[counter, j] = torch.from_numpy(data2_save[j])

        else:

            idx = get_nw_idx_single(nw_norm, nw_plot_bins)
            f = scipy.interpolate.interp1d(phi_norm, eta_norm, bounds_error=False, fill_value=0, kind='linear')
            f131 = scipy.interpolate.interp1d(phi_norm, eta_131_norm, bounds_error=False, fill_value=0, kind='linear')
            f2 = scipy.interpolate.interp1d(phi_norm, eta_2_norm, bounds_error=False, fill_value=0, kind='linear')

            eta_new = f(phi_plot_bins)
            eta131_new = f131(phi_plot_bins)
            eta2_new = f2(phi_plot_bins)

            X[counter, idx] = torch.from_numpy(eta_new)
            X_131[counter, idx] = torch.from_numpy(eta131_new)
            X_2[counter, idx] = torch.from_numpy(eta2_new)

        labels[counter] = torch.tensor(data_slice_params.iloc[0].values)

        X_stack = torch.stack((X, X_131, X_2), dim=1)

        counter = counter + 1

    X = unnorm_pe(X)
    X[X==1] = 0

    y = normalize_params(labels)
    #y2 = y[:,0:2]

    return X_stack, y, df_create, X

def main():

    resolution = (96, 96)
    batch_size = 27

    path_read = "exp_data/"
    df = pd.read_csv(f"{path_read}import-exp-data.csv")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model = InceptionV1()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),
        lr=0.001, weight_decay=0)

    #loss_fn = loss_functions.CustomMSELoss1()
    loss_fn = torch.nn.MSELoss()

    checkpoint = torch.load("../model_best_accuracy_chkpt_3_inputv2.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    X, y, table, eta_sp = load_exp_data(df, device, resolution, batch_size)
    y = y.to(device)
    X = X.to(device)
    model.eval()
    
    avg_loss = 0
    avg_error = 0

    with torch.no_grad():
        pred = model(X)
        pred_unnorm = mike.unnormalize_B_params(pred)
        Bg = pred_unnorm[:,0]
        Bth = pred_unnorm[:,1]
        pred_new = torch.clone(pred)
        Pe = ryan.recalc_pe_prediction(device, Bg, Bth, eta_sp, resolution, batch_size).to(device)
        pred_new = torch.column_stack((pred_new, torch.zeros(batch_size,1).to(device)))
        pred_new[:,2] = mike.normalize_Pe(Pe.squeeze())
        loss = loss_fn(pred, y[:,0:2])
        avg_loss += loss
        avg_error += torch.mean(torch.abs(y - pred_new) / y, 0)

        #print(pred, pred_new, y)

    print(f'avg_loss\tavg_error[0]\tavg_error[1]\tavg_error[2]')
    print(f'{avg_loss}\t{avg_error[0]}\t{avg_error[1]}\t{avg_error[2]}')

    bg_true, bth_true, pe_true = mike.unnormalize_params(y)        
    bg_pred, bth_pred, pe_pred = mike.unnormalize_params(pred_new)

    y_unnorm = y.detach().clone()
    pred_unnorm_data = pred_new.detach().clone()
    y_unnorm[:,0] = bg_true
    y_unnorm[:,1] = bth_true
    y_unnorm[:,2] = pe_true

    pred_unnorm_data[:,0] = bg_pred
    pred_unnorm_data[:,1] = bth_pred
    pred_unnorm_data[:,2] = pe_pred

    table["True Bg"] = bg_true.cpu()
    table["Pred Bg"] = bg_pred.cpu()
    table["True Bth"] = bth_true.cpu()
    table["Pred Bth"] = bth_pred.cpu()
    table["True Pe"] = pe_true.cpu()
    table["Pred Pe"] = pe_pred.cpu()

    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):  # more options can be specified
        print(table)
    plot_eval.plot_data(y_unnorm, pred_unnorm_data)

    plot_eval.record_fits(df, pred_unnorm_data, eta_sp, batch_size, device)

   
if __name__ == '__main__':
    main()
