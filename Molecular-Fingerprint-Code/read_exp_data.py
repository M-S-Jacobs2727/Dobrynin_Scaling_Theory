import numpy as np
import pandas as pd
import collections
import scipy.interpolate
import generate_surfaces

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
    Reads in experimental datasets collected from our previous works. Obtains interpolated 32x32 grid data and outputs each system to a text file in a designated folder. Another function will be created to take in a general input in the form (Nw, phi, eta_sp)

        Arguments:
            df (DataFrame) Data regarding all of the data from papers used in our papers
            path_save (string) Folder where output files are saved
            bins (dict) Bin data obtained from surface_bins.json

        Output:
            None
    """
    features = ['Nw', 'phi', 'eta_sp']

    df2 = df.copy()

    # log before normalizing (nw, phi, eta_sp) space
    df2[features] = np.log10(df2[features])

    # normalize (nw, phi, eta_sp) space to (0, 1)
    df2['Nw'] = (df2['Nw']-np.log10(bins['Nw'].min))/(np.log10(bins['Nw'].max)-np.log10(bins['Nw'].min))
    df2['phi'] = (df2['phi']-np.log10(bins['phi'].min))/(np.log10(bins['phi'].max)-np.log10(bins['phi'].min))
    df2['eta_sp'] = (df2['eta_sp']-np.log10(bins['eta_sp'].min))/(np.log10(bins['eta_sp'].max)-np.log10(bins['eta_sp'].min))

    for i in np.unique(df['group']):

        data_slice = df2[df2['group']==i][features]

        x = data_slice['Nw']
        y = data_slice['phi']
        z = data_slice['eta_sp']

        xplotv = np.linspace(0,1,bins['Nw'].num_bins)
        yplotv = np.linspace(0,1,bins['phi'].num_bins)
        xplot, yplot = np.meshgrid(xplotv, yplotv)

        if len(np.unique(data_slice['Nw'])) > 1:

            zgriddata = scipy.interpolate.griddata(np.array([x.ravel(),y.ravel()]).T,z.ravel(),np.array([xplot.ravel(),yplot.ravel()]).T, method='linear', fill_value=0)
            data_save = zgriddata.copy()
            data_save = data_save.reshape(bins['phi'].num_bins,bins['Nw'].num_bins)

        else:
            f = scipy.interpolate.interp1d(y,z, bounds_error=False, fill_value=0)
            znew = f(yplotv)
            nw_val = np.unique(x)
            zgriddata = get_grid_single_nw_data(nw_val, xplotv, znew)
            data_save = zgriddata.reshape(bins['phi'].num_bins,bins['Nw'].num_bins)

        polymer = np.unique(df[df['group']==i]['Polymer'])[0]
        solvent = np.unique(df[df['group']==i]['Solvent'])[0]
        Bg = np.round(np.unique(df[df['group']==i]['Bg'])[0],2)
        Bth = np.round(np.unique(df[df['group']==i]['Bth'])[0],2)
        Pe = np.round(np.unique(df[df['group']==i]['Pe'])[0],2)
        np.savetxt(f"{path_save}{polymer}_{solvent}_{Bg}_{Bth}_{Pe}.txt", data_save)

def main():
    path_read = "exp_data\\"
    path_save = "exp_data_gen\\"
    bins = generate_surfaces.generate_surface_bins()
    df = pd.read_csv(f"{path_read}new-df.csv")
    read_exp_data(df, path_save, bins)

if __name__ == '__main__':
    main()