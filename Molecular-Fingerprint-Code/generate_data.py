import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

def create_curves(Bg, Bth, Pe, Nw):
    '''
    Generate a set of theoretical specific viscosity curves over a given input of weight-average molecular weight

    Arguments:
        Bg: (float) B-parameter in good regime
        Bth: (float) B-parameter in theta regime
        Pe: (float) Packing number
        Nw: (ndarray) Set of weight-average degrees of polymerization

    Returns:
        arr: (ndarray) Array of parameters in the form of [Bg, Bth, Pe, Nw, phi, eta_sp]   
    '''
    # get number of input Nw
    num_nw = len(Nw)
    # define number of data points for each nw
    num_data_points_one_nw = 16
    # define concentration threshold where anything past seems unreasonable (towards bulk polymer concentration)
    phi_star_threshold = 0.01
    # init output array, each nw will have x number of data points data points of (phi,eta_sp)
    arr = np.zeros((num_data_points_one_nw*num_nw,6))

    # calculate crossover concentrations
    phi_star = (Bg ** 3) * (Nw ** (1-3*0.588))
    phi_th = (Bth ** 3.0) * (Bth/Bg) ** (1/(2*0.588-1))
    phi_star_star = (Bth ** 4)
    # account for nw where semidilute solution regime starts after theta blob regime
    phi_star[phi_star > phi_th] = (Bth ** 3) * (Nw[phi_star > phi_th] ** (1-3*0.5))

    # generate phi space
    phi = np.geomspace(phi_star,phi_star_threshold,num_data_points_one_nw)

    # calculate g
    g_g = Bg**(3/(3*0.588-1))*(phi)**(1/(1-3*0.588))
    g_th = Bth**(3/(3*0.5-1))*(phi)**(1/(1-3*0.5))
    g = np.minimum(g_g,g_th)

    # calculate Ne
    Ne_g = Pe**2 * (Bg**3/phi)**(1/(3*0.588-1))
    Ne_th = Pe**2*Bth**6*phi**-2
    Ne = np.minimum(Ne_g, Ne_th)

    Nw_g = Nw / g

    # calculate eta_sp
    eta_sp_1 = Nw_g*(1+(Nw/Ne)**2)
    eta_sp_2 = Nw * (1+(Nw/Ne)**2)*phi/Bth**2
    eta_sp = np.minimum(eta_sp_1, eta_sp_2)

    # format nw, phi, eta_sp to return to array
    nw_data = np.sort(np.tile(Nw,num_data_points_one_nw))
    phi_data = np.concatenate((np.hsplit(phi.T,1)), axis=None)
    eta_sp_data = np.concatenate((np.hsplit(eta_sp.T,1)), axis=None)

    # return array of data
    arr[:,0] = Bg
    arr[:,1] = Bth
    arr[:,2] = Pe
    arr[:,3] = nw_data
    arr[:,4] = phi_data
    arr[:,5] = eta_sp_data

    # delete rows where phi > threshold or eta_sp < 1
    arr = np.delete(arr, np.where((arr[:,5]<1) | (arr[:,4]>0.01)),axis=0)

    # Apply 5% noise from gaussian distribution to phi and eta_sp data
    percentage = 0.05
    noise = np.random.normal(0,1, len(arr)) * percentage
    arr[:,4] = arr[:,4] + noise*arr[:,4]
    noise = np.random.normal(0,1, len(arr)) * percentage
    arr[:,5] = arr[:,5] + noise*arr[:,5]

    return arr

def generate_grid():
    '''
    Generate grid of B-parameters, Pe, and Nw data to use for data generation

    Arguments:
        None

    Returns:
        grid: (dict) Values for each parameter input]   
    '''
    grid = {"Bg" : np.round(np.linspace(0.3, 1.6, 131),2),  

    "Bth" : np.round(np.linspace(0.2, 0.9, 71),2), 

    "Pe" : np.round(np.linspace(2,20,181),1), 

    "Nw" : np.array(np.geomspace(10, 300000, 16).astype(int)),}
    
    return grid

# Main
def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    grid = generate_grid()
    col_names = ['Bg', 'Bth', 'Pe', 'Nw', 'phi', 'eta_sp']
    for a in grid['Bg']:
        for b in grid['Bth']:
            df = pd.DataFrame(columns = col_names)
            for c in grid['Pe']:
                arr = create_curves(a, b, c, grid['Nw'])
                df2 = pd.DataFrame(data=arr, columns=col_names)
                df = df.append(df2)
            df = df.reset_index(drop=True)
            df.to_csv(f"generated_data\dataset_{a}_{b}.csv")
        print(a)

if __name__ == '__main__':
    main()