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
`generators.py` and `data_processing.py` and follow this example:

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

    generator = generators.SurfaceGenerator(config, strip_nw=True)

    for surfaces, features in generator(num_batches=1000):
        ...

## `main.py`
Run this from the command line as

    python3 main.py <configuration_filename> [-l logfile] [-m modelfile] [-v] [-h]

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
`NNConfig`, the configuration class, initialized from the given configuration file.


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

## `loss_funcs.py`
Two custom loss functions, `LogCoshLoss` is taken from a stackexchange post (cite) and 
`CustomMSELoss` applies an MSE loss without punishing in the case of an athermal solvent
($B_{g} < B_{th}^{0.824}$). We have rejected `CustomMSELoss` in favor of splitting the
athermal condition into a separate model (see data_processing.Mode).

## `training.py`
Contains the train and test functions used to iterate the NN model in the main
module.

## `test_checkpoint_accuracy.py`
This loads a model and generates new surfaces to create parity plots for each of the 
evaluated features.

## `speed_test.py`
A simple script to test the speed of surface generation as a function of batch
size and resolution.

## Requires
Python >= 3.8
numpy >= 1.17.4
PyTorch >= 1.8
PyYAML >= 5.0