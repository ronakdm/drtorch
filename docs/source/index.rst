.. diropt documentation main file

Diropt
======

This library is built to perform distributionally robust optimization (DRO) within existing PyTorch workflows with a single additional line of code.
The focus is on spectral risk measure-based learning which is described in detail in `this AISTATS 2023 paper <https://proceedings.mlr.press/v206/mehta23b.html>`
and `this ICLR 2024 paper <https://arxiv.org/abs/2310.13863>`_ (Spotlight Presentation). Distributionally robust objectives apply a sample reweighting to the 
observerd training data within each mini-batch inorder to robustify models against distribution shifts that occur at test time. The main features are:

* Implementations of the superquantile/conditional value-at-risk (CVaR), extremile, and exponential spectral risk measures.
* Implementations of the pool adjacent violators (PAV) algorithm for computing regularized spectral risk measures with Kullback-Leibler and Chi-Squared divergences.

.. toctree::
  :maxdepth: 2
  :caption: Contents:
  :hidden:

  quickstart
  api

Installation
------------------

Once you have installed PyTorch >=1.6. (see `instructions <https://pytorch.org/get-started/locally/>`_), you can install Diropt by running

.. code-block:: bash

    $ git clone git@github.com:ronakdm/drtorch.git
    $ cd diropt
    $ pip install -e .

Quickstart
------------------

First, we construct a function that inputs a vector of losses and returns a probability distribution over elements in this loss vector.

.. code-block:: python

  >>> from diropt import make_spectral_risk_measure, make_extremile_spectrum
  >>> spectrum = make_superquantile_spectrum(batch_size, 2.0)
  >>> compute_sample_weight = make_spectral_risk_measure(spectrum, penalty="chi2", shift_cost=1.0)

Assume that we have computed a vector of losses based on a model output in PyTorch. We can then use the function above and back propagate through the weighted sum of losses.

.. code-block:: python

  >>> x, y = get_batch()
  >>> logits = model(x)
  >>> losses = torch.nn.functional.cross_entropy(logits, y, reduction="none")
  >>> with torch.no_grad():
  >>>     weights = compute_sample_weight(losses.cpu().numpy()).to(device)
  >>> loss = weights @ losses
  >>> loss.backward()

A detailed quickstart guide is given in `docs/source/quicstart.ipynb`, whereas an example training on Fashion MNIST is given in `examples/train_fashion_mnist.ipynb`.


Contributing
------------------

If you find any bugs, please raise an issue on GitHub.
If you would like to contribute, please submit a pull request.
We encourage and highly value community contributions.


Authors
------------------

This package is written and maintained by `Ronak Mehta <https://ronakdm.github.io/>`_.


Cite
------------------

If you find this package useful, or you use it in your research, please cite:

.. code-block::

    @inproceedings{mehta2023stochastic,
      title={{Stochastic Optimization for Spectral Risk Measures}},
      author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Liu, Lang and Harchaoui, Zaid},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      pages={10112--10159},
      year={2023},
      organization={PMLR}
    }

Acknowledgements
------------------

