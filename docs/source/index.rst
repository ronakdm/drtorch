Diropt
======

.. toctree::
  :maxdepth: 2
  :caption: Contents:
  :hidden:

  quickstart

  api


Installation
------------

You need to install PyTorch >=1.6., see `instructions <https://pytorch.org/get-started/locally/>`_.
Once pytorch is installed, you can install diropt by running


.. code-block:: bash

    $ git clone git@github.com:ronakdm/drtorch.git
    $ cd diropt
    $ pip install -e .


Documentation
-------------

To build the documentation, install the required dependencies using ``pip install -e .[docs]``.
Then place yourself in the folder ``docs`` and run ``make html``.


Contributing
-------------

If you find any bugs, please raise an issue on GitHub.
If you would like to contribute, please submit a pull request.
We encourage and highly value community contributions.


Authors
-------

This package is written and maintained by `Ronak Mehta <https://ronakdm.github.io/>`_.


Cite
----

If you find this package useful, or you use it in your research, please cite:

.. code-block::

    @inproceedings{mehta2023stochastic,
      title={Stochastic optimization for spectral risk measures},
      author={Mehta, Ronak and Roulet, Vincent and Pillutla, Krishna and Liu, Lang and Harchaoui, Zaid},
      booktitle={International Conference on Artificial Intelligence and Statistics},
      pages={10112--10159},
      year={2023},
      organization={PMLR}
    }

