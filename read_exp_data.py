import numpy as np
import pandas as pd
import collections
from scipy import interpolate
from generate_surfaces import generate_surface_bins

Bin = collections.namedtuple('Bin', ['min', 'max', 'num_bins'])

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

def read_exp_data(df, path_save, bins):
    """
    Reads in experimental datasets collected from our previous works. Obtains interpolated grid data and outputs each system to a text file in a designated folder. Another function will be created to take in a general input in the form (Nw, phi, eta_sp)

        Arguments:
            df (DataFrame) Data regarding all of the data from papers used in our papers
            path_save (string) Folder where output files are saved
            bins (dict) Bin data obtained from surface_bins.json

        Output:
            None
    """
    features = ['Nw', 'phi', 'eta_sp']

    df2 = df.copy()

    df2['Nw'] = (df2['Nw']-bins['Nw'].min)/(bins['Nw'].max-bins['Nw'].min)
    df2['phi'] = (df2['phi']-bins['phi'].min)/(bins['phi'].max-bins['phi'].min)
    df2['eta_sp'] = (df2['eta_sp']-bins['eta_sp'].min)/(bins['eta_sp'].max-bins['eta_sp'].min)

    for i in np.unique(df['group']):

        data_slice = df2[df2['group']==i][features]

        x = data_slice['Nw']
        y = data_slice['phi']
        z = data_slice['eta_sp']

        xplotv = np.linspace(0,1,bins['Nw'].num_bins)
        yplotv = np.linspace(0,1,bins['phi'].num_bins)
        xplot, yplot = np.meshgrid(xplotv, yplotv)

        if len(np.unique(data_slice['Nw'])) > 1:

        # If there's more than one Nw, output is all zeros
        # to fix
        # How to deal with more than 1 Nw in same bin? Currently generating data of 64 phi over 64 Nw vals for training
            zgriddata = interpolate.griddata(np.array([x.ravel(),y.ravel()]).T,z.ravel(),np.array([xplot.ravel(),yplot.ravel()]).T, method='cubic', fill_value=0)
            data_save = zgriddata.copy()

        else:
            f = interpolate.interp1d(y,z, fill_value='extrapolate')
            znew = f(yplotv)
            nw_val = np.unique(x)
            zgriddata = get_grid_single_nw_data(nw_val, xplotv, znew)
            data_save = zgriddata.reshape(bins['phi'].num_bins,bins['Nw'].num_bins)

        polymer = np.unique(df[df['group']==i]['Polymer'])[0]
        solvent = np.unique(df[df['group']==i]['Solvent'])[0]
        Bg = np.unique(df[df['group']==i]['Bg'])[0]
        Bth = np.unique(df[df['group']==i]['Bth'])[0]
        Pe = np.unique(df[df['group']==i]['Pe'])[0]
        np.savetxt(f"{path_save}{polymer}_{solvent}_{Bg}_{Bth}_{Pe}.txt", data_save)

def main():
    path_read = "exp_data\\"
    path_save = "exp_data_gen\\"
    bins = generate_surface_bins()
    df = pd.read_csv(f"{path_read}new-df.csv")
    read_exp_data(df, path_save, bins)

if __name__ == '__main__':
    main()