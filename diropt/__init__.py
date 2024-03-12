"""Diropt: Distributionally robust optimization in pytorch."""
from diropt._src.pav import l2_centered_isotonic_regression
from diropt._src.pav import neg_entropy_centered_isotonic_regression

from diropt._src.spectral_risk import make_spectral_risk_measure
from diropt._src.spectral_risk import spectral_risk_measure_maximization_oracle

from diropt._src.spectra import make_esrm_spectrum
from diropt._src.spectra import make_extremile_spectrum
from diropt._src.spectra import make_superquantile_spectrum

__version__ = "0.0.1.dev"

__all__ =[
  "l2_centered_isotonic_regression",
  "neg_entropy_centered_isotonic_regression", 
  "make_esrm_spectrum",
  "make_extremile_spectrum",
  "make_spectral_risk_measure",
  "make_superquantile_spectrum"
  "spectral_risk_measure_maximization_oracle",
]