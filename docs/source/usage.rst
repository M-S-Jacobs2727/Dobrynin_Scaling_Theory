Usage
=====

First, import both PyTorch and PSST:

.. code:: python

    >>> import torch
    >>> import psst

You can feel free to use your own model or use one we provide:

.. code:: python

    >>> from psst.models import Inception3, Vgg13

We recommend writing a YAML configuration file based on those provided in the examples.
Load it:

.. code:: python
    
    >>> run_config, adam_config, gen_config = psst.loadConfig("config.yaml")

and create the model, optimizer, and loss function:

.. code:: python

    >>> model = Inception3()
    >>> optim = torch.optim.Adam(model.parameters(), **adam_config)
    >>> loss_fn = torch.nn.MSELoss()

Next, create an instance of the ``SampleGenerator``:

.. code:: python

    >>> gen_samples = psst.SampleGenerator(
    ...     device=torch.device("cuda"),
    ...     **gen_config,
    ... )

and send everything into the ``train_model`` function to run the entire train/test
cycle:

.. code:: python

    >>> train_model(
    ...     model=model,
    ...     loss_fn=loss_fn,
    ...     optimizer=optim,
    ...     generator=gen_samples,
    ...     **run_config,
    ... )
