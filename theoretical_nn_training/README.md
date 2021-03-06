# Theoretical Neural Network Training
This code is used to train a neural network (simple, convolutional, or inception-block)
to predict values of $B_{g}$, $B_{th}$, and $P_{e}$ from the dependence of specific
viscosity $\eta_{sp}$ on the weight-average degree of polymerization $N_{w}$ and 
reduced concentration $\varphi=cl^{3}$ ($c$ is concentration in number of repeat units
per unit volume and $l$ is the repeat unit projection length).

The model is initially trained on fully analytically-generated data (see 
`generators.py`). Next, we train the model on more generated data, but with much fewer
values of $N_{w}$ to simulate more realistic conditions, where experimentalists only
sample a small number of molecular weights. Finally, the data is evaluated on 
experimental datasets that were previously analyzed by hand.

If you want to test the surface generators with custom code, just take the files
`generators.py`, `configuration.py` and `data_processing.py` and follow this example:

    import torch

    from data_processing import Resolution, Range, Mode
    import configuration
    import generators

    config = configuration.NNConfig(
        device = torch.device('cuda'),
        resolution = Resolution(256, 256),
        phi_range = Range(1e-6, 0.01),
        nw_range = Range(100, 1e6),
        eta_sp_range = Range(1, 1e7),
        bg_range = Range(0.3, 1.1),
        bth_range = Range(0.2, 0.7),
        pe_range = Range(4, 20),
        mode = Mode.MIXED,
        batch_size = 100,
        num_nw_strips = 0,
    )

    generator = generators.SurfaceGenerator(config)

    for surfaces, features in generator(num_batches=1000):
        ...

## Requirements
- Python >= 3.8
- numpy >= 1.17.4
- PyTorch >= 1.8
- PyYAML >= 5.0

## `main.py`
Run this from the command line as

    python3 main.py <configuration_filename> [-m modelfile] [-v] [-h]

where `<configuration_filename>` is a path to a YAML or JSON configuration file (see
examples in the configurations folder). This will create a model, train it on the
generated data, and save the results (model, optimizer, and losses and errors over
training iterations). If a model file is passed in with the -m argument, the model
and optimizer will be loaded from the given file as a dictionary with keys 'model' and
'optimizer'.

## `configurations/` and `configuration.py`
Sample configurations that define the model, training, and testing are defined in YAML 
or JSON files in the configurations directory. One of these is passed as the first
command line argument into the main module. The `configuration.py` module defines 
`NNConfig`, the configuration class, and the `read_config_from_file` function, to
initialize a config from the given configuration file.


## `generators.py`
This file contains two efficient generator classes based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by a subset of the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. Each of these is sampled from a choice of `Uniform`, 
`LogNormal`, or `Beta` distributions. To return partial surfaces with only a narrow 
range of $N_w$ values, use the optional argument `strip_nw=True` when initializing the
generator.

## `data_processing.py`
This is somewhat of a catch-all for classes and functions that are used throughout the
codebase. `Mode` indicates what subset of the features $\{B_{g}, B_{th}, P_{e}\}$ is
being generated; `Resolution` is a dataclass for the resolution of the
generated surfaces, either 2D or 3D; `Range` defines the minimum, maximum, and 
distributions for various parameters; and `normalize_params`, `unnormalize_params`,
`preprocess_eta_sp`, etc. perform basic normalization operations on the data and 
features.

## `models.py`
Three flexible classes for neural networks: a fully-connected dense network, a 2D 
convolutional neural network, and a 3D convolutional neural network.

## `training.py`
Contains the train and test functions used to iterate the NN model in the main
module.