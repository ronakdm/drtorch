📖 API
======

.. currentmodule:: diropt

.. autosummary::
  l2_centered_isotonic_regression
  neg_entropy_centered_isotonic_regression
  make_esrm_spectrum
  make_extremile_spectrum
  make_spectral_risk_measure
  make_superquantile_spectrum
  spectral_risk_measure_maximization_oracle


Create risk measure
-------------------
.. autofunction:: make_spectral_risk_measure
.. autofunction:: spectral_risk_measure_maximization_oracle


Dual maximization oracles
-------------------------

Pool adjacent violator algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: l2_centered_isotonic_regression
.. autofunction:: neg_entropy_centered_isotonic_regression


Spectrums
---------

Extremile
~~~~~~~~~
.. autofunction:: make_extremile_spectrum

Superquantile
~~~~~~~~~~~~~
.. autofunction:: make_superquantile_spectrum

Exponential spectral risk measure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: make_esrm_spectrum



