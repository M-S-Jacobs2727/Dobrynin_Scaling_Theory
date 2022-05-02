import collections
import json
import numba
import numpy as np
import multiprocessing as mpr

Param = collections.namedtuple('Param', ('min', 'max'))

class Processor:
    """Before use in the neural network, the generated data should have noise
    applied, then be normalized and capped.
    
    Normalization:
    Each parameter `(Bg, Bth, Pe)` and variable `(phi, Nw, eta_sp)` must be
    normalized before being used in the neural net. The variables are first
    reduced by taking the logarithm (base e). Then, parameters and variables
    are linearly reduced to a range of [0, 1] with the map 
    `x -> (x - min(x)) / (max(x) - min(x))`
    and back with
    `y -> min(x) + y * (max(x) - min(x))`
    where `min(x)` and `max(x)` are fixed quantities defined in json files
    passed in as arguments to the class.
    """
    def __init__(self, param_file, data_file):
        # Parameters Bg, Bth, Pe
        with open(param_file) as f:
            param_ranges = json.load(f)
        
        self.Bg = Param(
            param_ranges['Bg_min'], 
            param_ranges['Bg_max']
        )
        self.Bth = Param(
            param_ranges['Bth_min'], 
            param_ranges['Bth_max']
        )
        self.Pe = Param(
            param_ranges['Pe_min'], 
            param_ranges['Pe_max']
        )

        # Data phi, Nw, eta_sp
        with open(data_file) as f:
            data_ranges = json.load(f)
        
        self.phi = Param(
            np.log(data_ranges['phi_min_bin']), 
            np.log(data_ranges['phi_max_bin'])
        )
        self.Nw = Param(
            np.log(data_ranges['Nw_min_bin']), 
            np.log(data_ranges['Nw_max_bin'])
        )
        self.eta_sp = Param(
            np.log(data_ranges['eta_sp_min_bin']), 
            np.log(data_ranges['eta_sp_max_bin'])
        )
    
    def add_noise(self, eta_sp, noise_pctge=5):
        eta_sp += np.random.normal(0, 1, eta_sp.shape) * eta_sp * 0.01 * noise_pctge
        return eta_sp

    def cap(self, eta_sp):
        """Limit eta_sp to range [self.eta_sp.min, self.eta_sp.max]
        """
        eta_sp[eta_sp < 0] = 0
        eta_sp[eta_sp > 1] = 0 # All OOB values are set to min
        return eta_sp

    def normalize_params(self, Bg, Bth, Pe):
        Bg = (Bg - self.Bg.min) / (self.Bg.max - self.Bg.min)
        Bth = (Bth - self.Bth.min) / (self.Bth.max - self.Bth.min)
        Pe = (Pe - self.Pe.min) / (self.Pe.max - self.Pe.min)
        return Bg, Bth, Pe

    def normalize_visc(self, eta_sp):
        eta_sp = (np.log(eta_sp) - self.eta_sp.min) / (self.eta_sp.max - self.eta_sp.min)
        return eta_sp
        
    def normalize_data(self, phi, Nw, eta_sp):
        phi = (np.log(phi) - self.phi.min) / (self.phi.max - self.phi.min)
        Nw = (np.log(Nw) - self.Nw.min) / (self.Nw.max - self.Nw.min)
        eta_sp = (np.log(eta_sp) - self.eta_sp.min) / (self.eta_sp.max - self.eta_sp.min)
        return phi, Nw, eta_sp
    
    def unnormalize_params(self, Bg, Bth, Pe):
        Bg = Bg * (self.Bg.max - self.Bg.min) + self.Bg.min
        Bth = Bth * (self.Bth.max - self.Bth.min) + self.Bth.min
        Pe = Pe * (self.Pe.max - self.Pe.min) + self.Pe.min
        return Bg, Bth, Pe
   
    def unnormalize_visc(self, eta_sp):
        eta_sp = eta_sp * (self.eta_sp.max - self.eta_sp.min) + self.eta_sp.min
        return np.exp(eta_sp)
   
    def unnormalize_data(self, phi, Nw, eta_sp):
        phi = phi * (self.phi.max - self.phi.min) + self.phi.min
        Nw = Nw * (self.Nw.max - self.Nw.min) + self.Nw.min
        eta_sp = eta_sp * (self.eta_sp.max - self.eta_sp.min) + self.eta_sp.min
        return np.exp([phi, Nw, eta_sp])

class SurfaceGenerator:
    def __init__(self, data_file):
        with open(data_file) as f:
            ranges = json.load(f)
        
        self.phi = np.geomspace(
            ranges['phi_min_bin'],
            ranges['phi_max_bin'], 
            ranges['phi_num_bins'], 
            endpoint=True
        )
        
        self.Nw = np.geomspace(
            ranges['Nw_min_bin'],
            ranges['Nw_max_bin'], 
            ranges['Nw_num_bins'], 
            endpoint=True
        )

        self.phi, self.Nw = np.meshgrid(self.phi, self.Nw)

    def generate(self, Bg : np.ndarray, Bth : np.ndarray, Pe : np.ndarray):
        """Input:
            Bg, Bth, Pe : 1-dimensional numpy arrays of equal length.
        
        Output:
            phi : Numpy array with shape (32, 32) of concentration values
            Nw  : Numpy array with shape (32, 32) of chain length values
            eta_sp : Numpy array with shape (Bg.shape[0], 32, 32) of viscosity 
                values, where the first dimension corresponds to the different
                values of Bg, Bth, and Pe.
        """
        if not (Bg.shape == Bth.shape == Pe.shape):
            raise ValueError('Arguments `Bg`, `Bth`, and `Pe` should have'
                ' identical shapes. Instead, they have shapes'
                f' {Bg.shape} {Bth.shape} {Pe.shape}.'
            )
        if np.ndim(Bg) > 1:
            raise ValueError('Arguments `Bg`, `Bth`, and `Pe` should have'
                f' one dimension. Instead, they have {np.ndim(Bg)}.'
            )

        if Bg.shape == (1,):
            eta_sp = fast_gen_surface(self.phi, self.Nw, Bg, Bth, Pe)
            eta_sp = np.array([eta_sp])
        else:
            with mpr.Pool(min(mpr.cpu_count() // 4, Bg.shape[0])) as pool:
                iters = pool.starmap(
                    fast_gen_surface, 
                    [(self.phi, self.Nw, b1, b2, b3) for b1, b2, b3 in 
                        zip(Bg, Bth, Pe)
                    ]
                )
                eta_sp = np.array([i for i in iters])
           
        return eta_sp

@numba.njit
def fast_gen_surface(phi : float, Nw : float, 
        Bg : float, Bth : float, Pe : float):
    """Generate `eta_sp` surface using the domain `(phi, Nw)` and parameters
    `Bg`, `Bth`, and `Pe`.
    """

    # Only defined for c < c**
    # Minimum accounts for crossover at c = c_th
    g = np.minimum(
        Bg**(3/0.764) / phi**(1/0.764),
        Bth**6 / phi**2
    )

    # Universal definition of Ne accounts for both 
    # Kavassalis-Noolandi and Rubinstein-Colby scaling
    Ne = Pe**2 * g * np.minimum(
        1, np.minimum(
            (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
            Bth**4 * phi**(2/3)
        )
    )
    # Minimum accounts for crossover at c = c**
    eta_sp = Nw * (1 + (Nw / Ne)**2) * np.minimum(
        1/g,
        phi / Bth**2
    )

    return eta_sp

def main():
    """For testing only.
    """
    surf = SurfaceGenerator('Molecular-Fingerprint-Code/surface_bins.json')
    print(surf.generate(np.array([0.8]), np.array([0.5]), np.array([10])))

if __name__ == '__main__':
    main()