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

## `main.py`
Run this from the command line as

    python3 main.py <configuration_filename> <output_directory>

where `<configuration_filename>` is a path to a YAML or JSON configuration file (see
examples in the configurations folder). This will create a model, train it on the
generated data, and save the results (model, optimizer, and losses and errors over
training iterations) in the output directory.

## `configuration.py` and `configurations/`
Sample configurations that define the model, training, and testing are defined in YAML 
or JSON files in the configurations directory. One of these is passed as the first
command line argument into the main module. The `configuration.py` module defines 
`NNConfig`, the configuration class, initialized from the given configuration file.


## `generators.py`
This file contains two efficient generator functions based on PyTorch to quickly
generate $(\varphi, N_{w}, \eta_{sp})$ surfaces defined by the parameter set
$\{B_{g}, B_{th}, P_{e}\}$. The parameter set is sampled from a choice of `Uniform`, 
`LogNormal`, or `Beta` distributions.

## `data_processing.py`
This is somewhat of a catch-all for classes and functions that are used throughout the
codebase. In particular, `Resolution` is a dataclass for the resolution of the
generated surfaces, either 2D or 3D; `Range` defines the minimum, maximum, and 
distributions for various parameters; and 

## `models.py`
Three flexible classes for neural networks.

## `loss_funcs.py`
Two custom loss functions, `LogCoshLoss` is taken from a stackexchange post (cite) and 
`CustomMSELoss` applies an MSE loss without punishing in the case of an athermal solvent
($B_{g} < B_{th}^{0.824}$).

## `training.py`

## `tuning.py`

## `test_checkpoint_accuracy.py`

## `speed_test.py`