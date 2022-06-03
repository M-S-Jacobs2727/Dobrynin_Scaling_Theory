import numpy as np
import pandas as pd
import collections
import scipy.interpolate
import torch
import json

Bin_X = collections.namedtuple('Bin_X', ['min', 'max', 'num_bins'])
Bin_y =collections.namedtuple('Bin_y', ['min', 'max'])

def generate_bin_params():
    """
    Read json file "surface_bins.json" to output bins to create (nw, phi, eta_sp) space
    
        Arguments:
            None

        Returns:
            Parameters: (dict) Dictionary of tuples. Number of bins [0], minimum bin [1] and maximum bin [2] for each coordinate of Nw, phi, and eta_sp
    """

    with open('exp_data/surface_bins.json') as f:
        bin_data_X = json.load(f)

    bins_X = {}
    bins_X['Nw'] = Bin_X(bin_data_X['Nw_min_bin'], bin_data_X['Nw_max_bin'], bin_data_X['Nw_num_bins'])
    bins_X['phi'] = Bin_X(bin_data_X['phi_min_bin'], bin_data_X['phi_max_bin'], bin_data_X['phi_num_bins'])
    bins_X['eta_sp'] = Bin_X(bin_data_X['eta_sp_min_bin'], bin_data_X['eta_sp_max_bin'], bin_data_X['eta_sp_num_bins'])

    with open('exp_data/Bg_Bth_Pe_range.json') as f:
        bin_data_y = json.load(f)

    bins_y = {}
    bins_y['Bg'] = Bin_y(bin_data_y['Bg_min'], bin_data_y['Bg_max'])
    bins_y['Bth'] = Bin_y(bin_data_y['Bth_min'], bin_data_y['Bth_max'])
    bins_y['Pe'] = Bin_y(bin_data_y['Pe_min'], bin_data_y['Pe_max'])

    return bins_X, bins_y

def get_grid_single_nw_data(nw_val, xplotv, z):
    """
    Return a grid with values in the column corresponding to the bin where Nw is located. All other columns are filled with zeros.

        Arguments:
            nw_val (int) Value of Nw of the experimental dataset
            xplotv (ndarray) Bins corresponding to each increment of normalized Nw
            z (Series) Values of interpolated normalized eta_sp (from interp1d) in a single system

        Returns:
            zgrid (ndarray) Grid of one row of interpolated normalized eta_sp values, all other values are zeros
    """


    zgrid = np.zeros([len(xplotv),len(xplotv)])
    idx = (np.abs(xplotv-nw_val)).argmin()
    zgrid[:,idx] = z

    return zgrid

def normalize_labels(y, bins_y):
    """Simple linear normalization.
    """
    #Bg_min = 0.30
    #Bg_max = 1.60
    #Bth_min = 0.20
    #Bth_max = 0.90
    #Pe_min = 2.0
    #Pe_max = 20.0

    y[:, 0] = (y[:, 0] - bins_y['Bg'].min)/(bins_y['Bg'].max-bins_y['Bg'].min)
    y[:, 1] = (y[:, 1] - bins_y['Bth'].min)/(bins_y['Bth'].max-bins_y['Bth'].min)
    y[:, 2] = (y[:, 2] - bins_y['Pe'].min)/(bins_y['Pe'].max-bins_y['Pe'].min)
    return y

#def read_exp_data(df, path_save, bins):
def read_exp_data(fname):
    """
    Reads in experimental datasets collected from our previous works. Obtains interpolated 32x32 grid data and outputs each system to a text file in a designated folder. Another function will be created to take in a general input in the form (Nw, phi, eta_sp)

        Arguments:
            df (DataFrame) Data regarding all of the data from papers used in our papers
            path_save (string) Folder where output files are saved
            bins (dict) Bin data obtained from surface_bins.json

        Output:
            None
    """

    bins_X, bins_y = generate_bin_params()

    features = ['Nw', 'phi', 'eta_sp']

    df = pd.read_csv(fname)

    df2 = df.copy()
    #df2 = df2[(df2['Bth']!= 0) & (df2['Bg'] != 0)]
    # log before normalizing (nw, phi, eta_sp) space
    df2[features] = np.log10(df2[features])

    # normalize (nw, phi, eta_sp) space to (0, 1)
    df2['Nw'] = (df2['Nw']-np.log10(bins_X['Nw'].min))/(np.log10(bins_X['Nw'].max)-np.log10(bins_X['Nw'].min))
    df2['phi'] = (df2['phi']-np.log10(bins_X['phi'].min))/(np.log10(bins_X['phi'].max)-np.log10(bins_X['phi'].min))
    df2['eta_sp'] = (df2['eta_sp']-np.log10(bins_X['eta_sp'].min))/(np.log10(bins_X['eta_sp'].max)-np.log10(bins_X['eta_sp'].min))
    counter = 0

    label_output = torch.zeros(len(np.unique(df2['group'])),3)
    #x_data = torch.zeros(bins_X['Nw'].num_bins, bins_X['phi'].num_bins)

    for i in np.unique(df2['group']):
        
        data_slice = df2[df2['group']==i][features]

        x = data_slice['Nw']
        y = data_slice['phi']
        z = data_slice['eta_sp']

        xplotv = np.linspace(0,1,bins_X['Nw'].num_bins)
        yplotv = np.linspace(0,1,bins_X['phi'].num_bins)
        xplot, yplot = np.meshgrid(xplotv, yplotv)

        if len(np.unique(data_slice['Nw'])) > 1:

            zgriddata = scipy.interpolate.griddata(np.array([x.ravel(),y.ravel()]).T,z.ravel(),np.array([xplot.ravel(),yplot.ravel()]).T, method='linear', fill_value=0)
            data_save = zgriddata.copy()
            data_save = data_save.reshape(bins_X['phi'].num_bins,bins_X['Nw'].num_bins)
            torch_data = torch.from_numpy(data_save)
            if counter==0:
                x_data = torch_data
            else:
                x_data = torch.cat((x_data, torch_data), 0)

        else:
            f = scipy.interpolate.interp1d(y,z, bounds_error=False, fill_value=0)
            znew = f(yplotv)
            nw_val = np.unique(x)
            zgriddata = get_grid_single_nw_data(nw_val, xplotv, znew)
            data_save = zgriddata.reshape(bins_X['phi'].num_bins,bins_X['Nw'].num_bins)
            torch_data = torch.from_numpy(data_save)
            if counter==0:
                x_data = torch_data
            else:
                x_data = torch.cat((x_data, torch_data), 0)

        #row = 128
        #x_data = x_data[torch.arange(1, x_data.shape[0]+1) >= row, ...]

        #polymer = np.unique(df[df['group']==i]['Polymer'])[0]
        #solvent = np.unique(df[df['group']==i]['Solvent'])[0]
        Bg = np.unique(df[df['group']==i]['Bg'])[0]
        Bth = np.unique(df[df['group']==i]['Bth'])[0]
        Pe = np.unique(df[df['group']==i]['Pe'])[0]
        #np.savetxt(f"{path_save}{counter}_res64_{Bg}_{Bth}_{Pe}_surface.txt", data_save)
        label_output[counter,0] = Bg
        label_output[counter,1] = Bth
        label_output[counter,2] = Pe
        counter = counter + 1

    labels_norm = normalize_labels(label_output, bins_y)
    #torch.save(labels_norm, "labels_normalized.pt")
    num_samples = int(x_data.size()[0]/x_data.size()[1])
    return x_data, labels_norm, num_samples
'''
def main():
    fname = "exp_data/data.csv"
    X, y, num_samples = read_exp_data(fname)
    x_np = X.numpy()
    np.savetxt("test_X_data", x_np)
    #x_df = pd.DataFrame(x_np)
    #x_df.to_csv('test_X_csv.csv')
    y_np = y.numpy()
    y_df = pd.DataFrame(y_np)
    y_df.to_csv('test_y_csv.csv')
    print(int(X.size()[0]/X.size()[1]))

if __name__ == '__main__':
    main()
'''
