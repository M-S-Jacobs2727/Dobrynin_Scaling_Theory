import numpy as np
import pandas as pd
import torch
import collections
from inceptionv3 import Inception3
from vgg13 import VGG13_Net
import scaling_torch_lib as mike
import warnings
from pandas.core.common import SettingWithCopyWarning
import recalc_pe as ryan
import output_tables as output
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(3e-5, 2e-2)
NW = Param(100, 1e5)
ETA_SP = Param(1, 1e6)

BG = Param(0.36, 1.55)
BTH = Param(0.22, 0.82)
PE = Param(2.5, 13.5)

ETA_SP_131 = Param(ETA_SP.min/NW.max/PHI.max**(1/(3*0.588-1)),ETA_SP.max/NW.min/PHI.min**(1/(3*0.588-1)))
ETA_SP_2 = Param(ETA_SP.min/NW.max/PHI.max**2,ETA_SP.max/NW.min/PHI.min**2)

NUM_BIN = 224

def get_grid_single_nw_data(nw_norm_val, nw_plot_bins, eta_new):

    eta_grid = np.zeros([len(nw_plot_bins), len(nw_plot_bins)])
    idx = np.abs(nw_plot_bins-nw_norm_val).argmin()
    eta_grid[:,idx] = eta_new
    return eta_grid

def get_nw_idx_single(nw_norm, nw_plot_bins):

    nw_norm_uniq = nw_norm.unique()  
    idx = np.abs(nw_plot_bins - nw_norm_uniq).argmin()

    return idx

def get_phi_idx_single(phi_norm, phi_plot_bins):

    phi_norm = phi_norm.to_numpy()

    phi_grid = np.zeros(len(phi_norm))

    for i in range(0, len(phi_norm)):

        idx_val = np.abs(phi_norm[i] - phi_plot_bins).argmin()
        phi_grid[i] = idx_val

    return phi_grid

def get_idx_multi(nw_norm, phi_norm, nw_plot_bins, phi_plot_bins):

    nw_idxs = np.zeros(len(nw_norm))
    phi_idxs = np.zeros(len(phi_norm))

    for i in range(0, len(nw_norm)):

        nw_idx_val = np.abs(nw_norm[i] - nw_plot_bins).argmin()
        phi_idx_val = np.abs(phi_norm[i] - phi_plot_bins).argmin()
        nw_idxs[i] = nw_idx_val
        phi_idxs[i] = phi_idx_val

    return nw_idxs, phi_idxs

def normalize_params(y):

    y_new = torch.clone(y)
    y_new[:,0] = (y_new[:,0]-BG.min) / (BG.max - BG.min)
    y_new[:,1] = (y_new[:,1]-PE.min) / (PE.max - PE.min) 
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

def unnorm_eta_sp(eta_sp_norm):

    eta_sp_unnorm = ETA_SP.min*torch.exp(eta_sp_norm*torch.log(torch.tensor(ETA_SP.max)/torch.tensor(ETA_SP.min)))
    return eta_sp_unnorm

def unnorm_eta_sp_131(eta_sp_131_norm):

    eta_sp_131_unnorm = ETA_SP_131.min*torch.exp(eta_sp_131_norm*torch.log(torch.tensor(ETA_SP_131.max)/torch.tensor(ETA_SP_131.min)))
    return eta_sp_131_unnorm

def unnorm_eta_sp_2(eta_sp_2_norm):

    eta_sp_2_unnorm = ETA_SP_2.min*torch.exp(eta_sp_2_norm*torch.log(torch.tensor(ETA_SP_2.max)/torch.tensor(ETA_SP_2.min)))
    return eta_sp_2_unnorm

def load_exp_data(df, device, resolution):

    features = ['Nw', 'phi', 'eta_sp', 'eta131', 'eta2']

    df2 = df.copy()

    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]
    df2["eta131"] = df2["eta_sp"] / df2["Nw"] / df2["phi"] ** (1 / (3 * 0.588 - 1))
    df2["eta2"] = df2["eta_sp"] / df2["Nw"] / df2["phi"] ** 2

    df2 = df2[(df2['Bg'] > 0) | (df2['Bth'] > 0)]

    for i in np.unique(df2['group']):

        if len(df2[df2['group']==i]) < 5:

            df2 = df2[df2.group != i]

    batch_size = len(np.unique(df2['group']))

    X = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN).to(device)
    X_131 = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN).to(device)
    X_2 = torch.zeros(len(np.unique(df2['group'])), NUM_BIN, NUM_BIN).to(device)

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

    for i in np.unique(df2['group']):

        data_slice = df2[df2['group']==i][features]

        nw_exp = data_slice['Nw']
        phi_exp = data_slice['phi']
        eta_exp = data_slice['eta_sp']
        eta131_exp = data_slice['eta131']
        eta2_exp = data_slice['eta2']

        nw_norm, phi_norm, eta_norm, eta_131_norm, eta_2_norm = normalize_data(nw_exp, phi_exp, eta_exp, eta131_exp, eta2_exp)

        # Retrieve Nw indices

        nw_plot_bins = np.linspace(0,1,NUM_BIN)
        phi_plot_bins = np.linspace(0,1,NUM_BIN)


        eta_norm = eta_norm.to_numpy()
        eta_131_norm = eta_131_norm.to_numpy()
        eta_2_norm = eta_2_norm.to_numpy()

        if len(np.unique(data_slice['Nw'])) > 1:

            nw_norm = nw_norm.to_numpy()
            phi_norm = phi_norm.to_numpy()

            nw_idx, phi_idx = get_idx_multi(nw_norm, phi_norm, nw_plot_bins,phi_plot_bins)        

            for j in range(0, len(phi_norm)):

                X[counter, int(nw_idx[j]), int(phi_idx[j])] = eta_norm[j]
                X_131[counter, int(nw_idx[j]), int(phi_idx[j])] = eta_131_norm[j]
                X_2[counter, int(nw_idx[j]), int(phi_idx[j])] = eta_2_norm[j]

        else:

            nw_idx = get_nw_idx_single(nw_norm, nw_plot_bins)
            phi_idx = get_phi_idx_single(phi_norm, phi_plot_bins)

            for j in range(0, len(phi_norm)):

                X[counter, nw_idx, int(phi_idx[j])] = eta_norm[j]
                X_131[counter, nw_idx, int(phi_idx[j])] = eta_131_norm[j]
                X_2[counter, nw_idx, int(phi_idx[j])] = eta_2_norm[j]

        counter = counter + 1

    X_131_norm = torch.unsqueeze(X_131, 1)
    X_2_norm = torch.unsqueeze(X_2, 1)

    return X_131_norm, X_2_norm

def get_all_systems_table(df, device):
    """Retrieves all systems that we are calculating Pe
    Input: 
        df: DataFrame of systems to analyze
        device: cuda or cpu object

    Output:
        table_data: Pandas DataFrame - table of Bg, Bth, Pe data for all systems that can be analyzed.
        bg_true: Bg true value labels on device
        bth_true: Bth true value labels on device
    """

    params = ['Bg', 'Bth', 'Pe', 'subgroup', 'group']

    # Get copy of data, filter all data within (Nw,phi,eta_sp) bounds
    # Also data must have at least a nonzero Bg or Bth
    df2 = df.copy()
    df2 = df2[(df2['Nw'] >= NW.min) & (df2['Nw'] <= NW.max)]
    df2 = df2[(df2['phi'] >= PHI.min) & (df2['phi'] <= PHI.max)]
    df2 = df2[(df2['eta_sp'] > ETA_SP.min) & (df2['eta_sp'] < ETA_SP.max)]
    df2 = df2[(df2['Bg'] > 0) | (df2['Bth'] > 0)]

    # We must have at least 5 data points in a subgroup to get a fit for Pe
    num_points_for_pe_fit = 5
    for i in np.unique(df2['group']):
        if len(df2[df2['group']==i]) < num_points_for_pe_fit:
            df2 = df2[df2.group != i]

    # Get sizes of bg_true set and bth_true set
    num_bg_points = len(np.unique(df2[df2["Bg"] > 0]["group"]))
    num_bth_points = len(np.unique(df2[df2["Bth"] > 0]["group"])) 

    # collect true values for Bg and Bth for each system
    bg_true = torch.zeros(num_bg_points,1).to(device)
    bth_true = torch.zeros(num_bth_points,1).to(device)

    # get number of systems that have a nonzero Bg or Bth
    num_points = len(np.unique(df2['group']))
    init = np.zeros(num_points)

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
                "Subgroup": pd.Series(data=init,dtype="int"),
                "Group": pd.Series(data=init,dtype="int")
             }
        )

    counter = 0
    bg_counter = 0
    bth_counter = 0

    for i in np.unique(df2['group']):

        data_slice_params = df2[df2['group']==i][params]
        bg_val = data_slice_params["Bg"].iloc[0]
        bth_val = data_slice_params["Bth"].iloc[0]
        pe_val = data_slice_params["Pe"].iloc[0]
        subgroup_val = data_slice_params["subgroup"].iloc[0]
        group_val = data_slice_params["group"].iloc[0]

        df_create["Polymer"][counter] = np.unique(df2[df2["group"] == i]["Polymer"])[0]
        df_create["Solvent"][counter] = np.unique(df2[df2["group"] == i]["Solvent"])[0]
        df_create["Num Nw"][counter] = len(np.unique(df2[df2["group"] == i]['Nw']))
        df_create["True Bg"][counter] = bg_val
        df_create["True Bth"][counter] = bth_val
        df_create["True Pe"][counter] = pe_val
        df_create["Subgroup"][counter] = subgroup_val
        df_create["Group"][counter] = group_val

        # get true values of bg and bth for each system
        if bg_val > 0:
            bg_true[bg_counter] = bg_val
            bg_counter += 1
        if bth_val > 0:
            bth_true[bth_counter] = bth_val
            bth_counter += 1

        counter += 1

    return df_create

def get_error_bg_bth(df):

    bg_set = ["True Bg", "Pred Bg"]
    bth_set = ["True Bth", "Pred Bth"]

    bg_data = df[bg_set]
    bth_data = df[bth_set]

    mask_bg = df["True Bg"] != 0
    mask_bth = df["True Bth"] != 0

    bg_error = np.average(np.abs(bg_data["True Bg"][mask_bg] - bg_data["Pred Bg"][mask_bg]) / bg_data["True Bg"][mask_bg] * 100)
    bth_error = np.average(np.abs(bth_data["True Bth"][mask_bth] - bth_data["Pred Bth"][mask_bth]) / bth_data["True Bth"][mask_bth] * 100)

    return bg_error, bth_error

def main():

    # read data, init model
    resolution = (224, 224)
    path_read = "/media/sayko/storage1/inceptionv3/test-output-data/exp-data/"
    df = pd.read_csv(f"{path_read}exp-data-clean.csv")
    device = torch.device('cpu')
    model = Inception3()
    model.to(device)

    # init data table of all systems
    table_vals = get_all_systems_table(df, device)

    # init Bg run

    checkpoint = torch.load("/media/sayko/storage1/CNN-Model-Analysis/Inception/Bg_inc_1_model_best_exp_error.pt",map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f'{epoch=}')

    X_131, X_2 = load_exp_data(df, device, resolution)
    X_131 = X_131.to(device)
    model.eval()
    
    avg_loss = 0

    with torch.no_grad():
        pred = model(X_131)

    bg_pred = mike.unnormalize_Bg_param(pred)

    table_vals["Pred Bg"] = bg_pred.tolist()

    checkpoint = torch.load("/media/sayko/storage1/CNN-Model-Analysis/Inception/Bth_inc_3_model_best_exp_error.pt",map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    X_2 = X_2.to(device)
    model.eval()

    avg_loss = 0
    with torch.no_grad():
        pred = model(X_2)

    bth_pred = mike.unnormalize_Bth_param(pred)

    table_vals["Pred Bth"] = bth_pred.tolist()
    bg_error, bth_error = get_error_bg_bth(table_vals)

    print(f'{bg_error=}, {bth_error=}')

    df_complete, pe_error = ryan.get_pe(table_vals, df, device)

    df_combo = ryan.get_pe_combo(df_complete, df, device) 

    df_g = ryan.get_pe_g(df_combo, df, device)

    df_th = ryan.get_pe_th(df_g, df, device)

    print(f'{pe_error=}')

    df_th.to_csv("inceptionv3-data-table.csv")

    output.state_diagram(df, df_th, device)
    output.universal_plot(df, df_th, device)
    df.to_csv("inceptionv3-df-input.csv")

if __name__ == '__main__':
    main()
