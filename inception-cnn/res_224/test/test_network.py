import numpy as np
import pandas as pd
import torch
import collections
import json
from googlenet import InceptionV1
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import scipy.interpolate
import loss_functions
import plot_eval
import scaling_torch_lib as mike
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(1e-6, 1e-2)
NW = Param(25, 2.3e5)
ETA_SP = Param(1, 3.1e6)

BG = Param(0, 1.53)
BTH = Param(0, 0.8)
PE = Param(0, 13.3)

ETA_SP_131 = Param(0.1, 140000)
ETA_SP_2 = Param(1, 4000000)

NUM_BIN = 224

def get_grid_single_nw_data(nw_norm_val, nw_plot_bins, eta_new):

    eta_grid = np.zeros([len(nw_plot_bins), len(nw_plot_bins)])
    idx = np.abs(nw_plot_bins-nw_norm_val).argmin()
    eta_grid[:,idx] = eta_new
    return eta_grid

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

    y[:,0] = y[:,0] / (BG.max - BG.min) - BG.min
    y[:,1] = y[:,1] / (BTH.max - BTH.min) - BTH.min
    y[:,2] = y[:,2] / (PE.max - PE.min) - PE.min
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

def load_exp_data(df, device):

    features = ['Nw', 'phi', 'eta_sp']
    params = ['Bg', 'Bth', 'Pe']

    df2 = df.copy()
   
    df2 = df2[(df2['Bg'] > 0) & (df2['Bth'] > 0)]

    num_points = len(np.unique(df2['group']))
    init = np.zeros(num_points)
    # Create data for y true vals
    labels = torch.zeros(len(np.unique(df2['group'])), 3)
    X = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN)
    X_131 = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN)
    X_2 = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN)

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

        num_points = len(np.unique(df2['group']))

        polymer = np.unique(df2[df2["group"] == i]["Polymer"])[0]
        solvent = np.unique(df2[df2["group"] == i]["Solvent"])[0]

        df_create["Polymer"][counter] = polymer
        df_create["Solvent"][counter] = solvent

        data_slice = df2[df2['group']==i][features]
        data_slice_params = df2[df2['group']==i][params]
        nw = data_slice['Nw']
        phi = data_slice['phi']
        eta = data_slice['eta_sp']

        # normalize
        eta_131 = eta/nw/phi**(1/(3*0.588-1))
        eta_2 = eta/nw/phi**2

        df_create["Num Nw"][counter] = len(np.unique(data_slice['Nw']))
        nw_norm, phi_norm, eta_norm, eta_131_norm, eta_2_norm = normalize_data(nw, phi, eta, eta_131, eta_2)

        nw_plot_bins = np.linspace(0,1,NUM_BIN)
        phi_plot_bins = np.linspace(0,1,NUM_BIN)
        nw_plot, phi_plot = np.meshgrid(nw_plot_bins, phi_plot_bins)

        if len(np.unique(data_slice['Nw'])) > 1:
            eta_grid_data = scipy.interpolate.griddata(np.array([nw_norm.ravel(),phi_norm.ravel()]).T,
                    eta_norm.ravel(),np.array([nw_plot.ravel(),phi_plot.ravel()]).T,
                    method='linear', fill_value=0)
            data_save = eta_grid_data.reshape(NUM_BIN,NUM_BIN)

            eta131_grid_data = scipy.interpolate.griddata(np.array([nw_norm.ravel(),phi_norm.ravel()]).T,
                    eta_131_norm.ravel(),np.array([nw_plot.ravel(),phi_plot.ravel()]).T,
                    method='linear', fill_value=0)
            data131_save = eta131_grid_data.reshape(NUM_BIN,NUM_BIN)

            eta2_grid_data = scipy.interpolate.griddata(np.array([nw_norm.ravel(),phi_norm.ravel()]).T,
                    eta_2_norm.ravel(),np.array([nw_plot.ravel(),phi_plot.ravel()]).T,
                    method='linear', fill_value=0)
            data2_save = eta2_grid_data.reshape(NUM_BIN,NUM_BIN)

        else:
            f = scipy.interpolate.interp1d(phi_norm, eta_norm, bounds_error=False, fill_value=0)
            f131 = scipy.interpolate.interp1d(phi_norm, eta_131_norm, bounds_error=False, fill_value=0)
            f2 = scipy.interpolate.interp1d(phi_norm, eta_2_norm, bounds_error=False, fill_value=0)
            eta_new = f(phi_plot_bins)
            eta131_new = f131(phi_plot_bins)
            eta2_new = f2(phi_plot_bins)
            nw_norm_val = np.unique(nw_norm)
            eta_grid_data = get_grid_single_nw_data(nw_norm_val, nw_plot_bins, eta_new)
            eta131_grid_data = get_grid_single_nw_data(nw_norm_val, nw_plot_bins, eta131_new)
            eta2_grid_data = get_grid_single_nw_data(nw_norm_val, nw_plot_bins, eta2_new)
            data_save = eta_grid_data.reshape(NUM_BIN,NUM_BIN)
            data131_save = eta131_grid_data.reshape(NUM_BIN,NUM_BIN)
            data2_save = eta2_grid_data.reshape(NUM_BIN,NUM_BIN)

        labels[counter] = torch.tensor(data_slice_params.iloc[0].values)
        X[counter] = torch.from_numpy(data_save)

        X_131[counter] = torch.from_numpy(data131_save)
        X_2[counter] = torch.from_numpy(data2_save)

        #xx = X[counter].flatten().numpy()
        
        X_stack = torch.stack((X_131, X_2), dim=1)

        counter = counter + 1

    y = normalize_params(labels)

    return X_stack, y, df_create

def main(d, model):

    path_read = "exp_data/"
    df = pd.read_csv(f"{path_read}import-exp-data.csv")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    model.to(d)

    torch.cuda.set_device(d)
    torch.distributed.init_process_group(backend='nccl', world_size=4, init_method=None, rank=d)

    model = DistributedDataParallel(model, device_ids=[d], output_device=d)

    optimizer = torch.optim.Adam(model.parameters(),
        lr=0.001, weight_decay=0)

    loss_fn = loss_functions.CustomMSELoss1()

    checkpoint = torch.load("model_best_accuracy_loss.pt")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    X, y, table = load_exp_data(df, device)
    y = y.to(device)

    if d==0:

        model.eval()
        
        avg_loss = 0
        avg_error = 0

        with torch.no_grad():
            pred = model(X)
            loss = loss_fn(pred, y)
            avg_loss += loss.item()
            #mask = (y[:,0] > 0) & (y[:,1] > 0)
            #avg_error += torch.mean(torch.abs(y[mask] - pred[mask]) / y[mask], 0)

            avg_error += torch.mean(torch.abs(y - pred) / y, 0)

        print(f'avg_loss\tavg_error[0]\tavg_error[1]\tavg_error[2]')
        print(f'{avg_loss}\t{avg_error[0]}\t{avg_error[1]}\t{avg_error[2]}')

        bg_true, bth_true, pe_true = mike.unnormalize_params(y)        
        bg_pred, bth_pred, pe_pred = mike.unnormalize_params(pred)

        y_unnorm = y.detach().clone()
        pred_unnorm = pred.detach().clone()
        y_unnorm[:,0] = bg_true
        y_unnorm[:,1] = bth_true
        y_unnorm[:,2] = pe_true

        pred_unnorm[:,0] = bg_pred
        pred_unnorm[:,1] = bth_pred
        pred_unnorm[:,2] = pe_pred

        table["True Bg"] = bg_true.cpu()
        table["Pred Bg"] = bg_pred.cpu()
        table["True Bth"] = bth_true.cpu()
        table["Pred Bth"] = bth_pred.cpu()
        table["True Pe"] = pe_true.cpu()
        table["Pred Pe"] = pe_pred.cpu()

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(table)

        plot_eval.plot_data(y, pred)


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 4, f"Requires at least 4 GPUs to run, but got {n_gpus}"
    model = InceptionV1()
    torch.multiprocessing.spawn(main, args=(model,), nprocs=4)
