.. psst documentation master file, created by
   sphinx-quickstart on Thu Oct 19 10:09:00 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PSST Library Documentation
================================

The Polymer Solution Scaling Theory (PSST) library is a PyTorch-based tool to train
neural networks to determine molecular parameters from specific viscosity data. It
does this by procedurally generating three molecular parameters, :math:`B_g`,
:math:`B_{th}`, and :math:`P_e`, computing the specific viscosity as a function of
repeat unit concentration and chain degree of polymerization, and training the network
on the normalized viscosity curve and the true values of the molecular parameters
(see the :doc:`theory` section for details).

.. toctree::
   :maxdepth: 3

   install
   usage
   api
   theory



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
