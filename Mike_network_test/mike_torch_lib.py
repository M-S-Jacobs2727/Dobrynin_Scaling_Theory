import collections
import json
import numba
import numpy as np
import multiprocessing as mpr
import torch

Param = collections.namedtuple('Param', ('min', 'max'))

def yield_surfaces(batch_size, num_batches, resolution=None, device=None):
    """Generate `batch_size` surfaces, based on ranges for `Bg`, `Bth`, and 
    `Pe`, to be used in a `for` loop. 

    It defines the resolution of the surface based on either user input 
    (keyword argument `resolution`) or the definitions in the file
    `surface_bins.json` (default). It then generates random values for `Bg`, 
    `Bth`, and `Pe`, evaluates the `(phi, Nw, eta_sp)` surface, and normalizes
    the result. The normalized values of `eta_sp` and `(Bg, Bth, Pe)` are 
    yielded as `X` and `y` for use in a neural network.

    Input:
        `batch_size` (`int`) : The length of the generated values.
        `resolution` (`tuple` of `int`s) : The shape of the last two dimensions
            of the generated values

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Generated 
            values of `eta_sp`.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated values of
            `(Bg, Bth, Pe)`.
    """
    with open('surface_bins.json') as f:
        ranges = json.load(f)
    
    if resolution is None:
        resolution = (ranges['phi_num_bins'], ranges['Nw_num_bins'])

    if device is None:
        device = torch.device('cuda')
    
    phi_min = ranges['phi_min_bin']
    phi_max = ranges['phi_max_bin']
    Nw_min = ranges['Nw_min_bin']
    Nw_max = ranges['Nw_max_bin']
    eta_sp_min = torch.tensor([ranges['eta_sp_min_bin']], device=device)
    eta_sp_max = torch.tensor([ranges['eta_sp_max_bin']], device=device)

    # Create tensors for phi (concentration) and Nw (chain length)
    phi = np.geomspace(
        phi_min,
        phi_max,
        resolution[0],
        endpoint=True
    )

    Nw = np.geomspace(
        Nw_min,
        Nw_max,
        resolution[1],
        endpoint=True
    )

    phi, Nw = np.meshgrid(phi, Nw)
    phi = torch.tile(torch.tensor(phi, device=device), (batch_size, 1, 1))
    Nw = torch.tile(torch.tensor(Nw, device=device), (batch_size, 1, 1))

    # Set up normalization for Bg, Bth, Pe
    with open('Bg_Bth_Pe_range.json') as f:
        ranges = json.load(f)
    
    Bg_min = ranges['Bg_min']
    Bg_max = ranges['Bg_max']
    Bth_min = ranges['Bth_min']
    Bth_max = ranges['Bth_max']
    Pe_min = ranges['Pe_min']
    Pe_max = ranges['Pe_max']

    def unnormalize_params(y):
        """Simple linear normalization.
        """
        Bg = y[:, 0] * (Bg_max - Bg_min) + Bg_min
        Bth = y[:, 1] * (Bth_max - Bth_min) + Bth_min
        Pe = y[:, 2] * (Pe_max - Pe_min) + Pe_min
        return Bg, Bth, Pe
    
    def normalize_visc(eta_sp):
        """Cap the values, then take the log, then normalize.
        """
        eta_sp = torch.fmin(eta_sp, torch.tensor([eta_sp_max], device=device))
        eta_sp = torch.fmax(eta_sp, torch.tensor([eta_sp_min], device=device))
        return (torch.log(eta_sp) - torch.log(eta_sp_min)) / \
            (torch.log(eta_sp_max) - torch.log(eta_sp_min))

    def generate_surfaces(Bg, Bth, Pe):
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)

        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        g = torch.fmin(
            Bg**(3/0.764) / phi**(1/0.764),
            Bth**6 / phi**2
        )

        # Universal definition of Ne accounts for both 
        # Kavassalis-Noolandi and Rubinstein-Colby scaling
        Ne = Pe**2 * g * torch.fmin(
            torch.tensor([1], device=device), torch.fmin(
                (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
                Bth**4 * phi**(2/3)
            )
        )

        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover at c = c**
        eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
            1/g,
            phi / Bth**2
        )

        return eta_sp

    for _ in range(num_batches):
        y = torch.rand((batch_size, 3), device=device, dtype=torch.float)
        Bg, Bth, Pe = unnormalize_params(y)
        eta_sp = generate_surfaces(Bg, Bth, Pe)
        X = normalize_visc(eta_sp).to(torch.float)
        yield X, y

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
        """Limit eta_sp to range [0, 1]. To be called after normalization.
        TODO: do this before normalization. self.eta_sp should not be logged
        ahead of time. This will avoid taking the log of a value in (-inf, 0].
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
    """Simple class with two attributes `self.phi` and `self.Nw`, which are
    meshgrids of concentration and strand length values, respectively.
    
    Probably shouldn't be a class, but a generator function, yielding a 
    batch of predetermined size upon a call of `next(generator)`.
    """
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
            eta_sp : Numpy array with shape 
                `(Bg.shape[0], self.phi.shape[0], self.Nw.shape[1])` of 
                viscosity values, where the first dimension corresponds to the 
                different values of Bg, Bth, and Pe.
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
    for i, surf in enumerate(yield_surfaces(100, 2, (64, 64))):
        X, y = surf
        print(X.size(), y.size())

if __name__ == '__main__':
    main()