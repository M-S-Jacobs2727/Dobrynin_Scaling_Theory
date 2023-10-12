# PSST (Polymer Solution Scaling Theory) Library

The `psst` library allows users to study and implement our deep learning scaling theory methods and test similar approaches.

## `psst` Module

The novel parts of the code are grouped into the module `psst`, which will be added to PyPI in the near future. For now, the library can be installed using the command `pip install .` in the head directory (the directory containing `pyproject.toml`). Dependencies will be handled by pip. Users may wish to create a new virtual environment first.

- The `models` submodule contains two convolutional neural network (CNN) models we used in our initial research, `models.Inception3` and `models.Vgg13`.
- `psst.configuration` defines the reading and structure of the system configuration, as specified in a YAML or JSON file.
- `psst.surface_generator` contains the `SurfaceGenerator` class that is used to procedurally generate viscosity curves as functions of concentration (`phi`) and chain degree of polymerization (`Nw`). The submodule also contains `normalize` and `unnormalize` functions. Normalizing transforms the true values, optionally on a log scale, to values between 0 and 1.
- `psst.training` contains the `train` and `validate` functions, as well as checkpointing functionality for the model and optimizer.

NOTE: The normalize/unnormalize functions and the checkpointing functionality may move to different submodules, perhaps new files.

## Other Directories

The `examples` directory contains scripts to optimize and train networks, and one to evaluate experimental data. These are similar to the scripts used during our research. Details are in `examples/README.md`.

The `deprecated` directory contains deprecated code that may be useful in the future.

The `doc` directory contains a markdown file of the mathematical derivation transforming the details in the original publications to the representations used in the code.

The `img` directory will contain images used in this README and in the `derivations.md` file.
