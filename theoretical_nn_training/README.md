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
`generators.py` and `data_processing.py`. Then follow the example:

    import torch

    import data_processing
    import generators

    config = generators.Config(
        device=torch.device('cuda'),
        resolution = data_processing.Resolution(256, 256),
        phi_range = data_processing.Range(1e-6, 0.01),
        nw_range = data_processing.Range(100, 1e6),
        eta_sp_range = data_processing.Range(1, 1e7),
        bg_range = data_processing.Range(0.3, 1.1),
        bth_range = data_processing.Range(0.2, 0.7),
        pe_range = data_processing.Range(4, 20),
        batch_size: 100,

    )

    generator = generators.SurfaceGenerator(config)

    for surfaces, features in generator(num_batches=1000):
        ...

## `main.py`
Run this from the command line as

    python3 main.py <configuration_filename> [-l logfile] [-v] [-h]

where `<configuration_filename>` is a path to a YAML or JSON configuration file (see
examples in the configurations folder). This will create a model, train it on the
generated data, and save the results (model, optimizer, and losses and errors over
training iterations).

## `configurations/` and `configuration.py`
Sample configurations that define the model, training, and testing are defined in YAML 
or JSON files in the configurations directory. One of these is passed as the first
command line argument into the main module. The `configuration.py` module defines 
`NNConfig`, the configuration class, initialized from the given configuration file.


## `generators.py`
This file contains two efficient generator clasees based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from a choice of `Uniform`, 
`LogNormal`, or `Beta` distributions.

## `data_processing.py`
This is somewhat of a catch-all for classes and functions that are used throughout the
codebase. `Resolution` is a dataclass for the resolution of the
generated surfaces, either 2D or 3D; `Range` defines the minimum, maximum, and 
distributions for various parameters; and `normalize_params`, `unnormalize_params`,
`preprocess_eta_sp`, etc. perform basic normalization operations on the data and 
features.

## `models.py`
Three flexible classes for neural networks.

## `loss_funcs.py`
Two custom loss functions, `LogCoshLoss` is taken from a stackexchange post (cite) and 
`CustomMSELoss` applies an MSE loss without punishing in the case of an athermal solvent
($B_{g} < B_{th}^{0.824}$).

## `training.py`
Contains the training and testing functions.

## `test_checkpoint_accuracy.py`

## `speed_test.py`
A simple script to test the speed of surface generation as a function of batch
size and resolution.