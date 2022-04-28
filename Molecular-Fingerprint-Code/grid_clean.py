import numpy as np

def consecutive(data, stepsize=1):
    '''
    Find sets with consecutive (increments of size stepsize) sequences

        Arguments:
            data: (ndarray) sequence of arrays
            stepsize: (int) value of stepsize that determines the sequence
        Returns:
            tuple of arrays that are in a sequence separated by stepsize. values not in a sequence are in their own array.
    '''
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def surface_edge(arr, arr_nn):

    '''
    Completes the surface plot of a given initial grid.

        Arguments:
            arr: (ndarray) grid created by griddata using the 'linear' method. (nw, phi) space with high nw and high phi are initiall zeros. The last row is all zeros, with some values in the second to last row between 0 and 1.
            arr_nn: (ndarray) grid created by griddata using the 'nearest' method.

        Outputs:
            arr: (ndarray) grid added by last row from arr_nn and space with high nw and phi are labeled as ones.

    '''

    # assign values in the last row where the second to last row (minus the last column) is between 0 and 1.
    arr[-1,np.where(arr[-2]>0)[-1][:-1]] = arr_nn[-1,np.where(arr[-2]>0)[-1][:-1]]
    # find all indices of (row, column) that are equal to zero
    idx = np.argwhere(arr==0)
    # find last sequence of columns that are zeros 
    cols = consecutive(np.where(arr[-1]==0)[-1])[-1]
    # find last sequence of rows that are zeros
    rows = consecutive(np.where(arr[:,-1]==0)[-1])[-1]
    # return last sequence of rows equal to zero
    masked_rows_idx = idx[np.isin(idx[:,0], rows)]
    # return last squence of columns equal to zero
    zeros_area = masked_rows_idx[np.isin(masked_rows_idx[:,1], cols)]
    # assign the sequences of rows and columns left values of 1
    arr[zeros_area[:,0], zeros_area[:,1]] = 1

    return arr