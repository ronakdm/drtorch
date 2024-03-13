"""Library of common spectra."""

import math
import numpy as np

def make_superquantile_spectrum(batch_size: int, tail_prob: float):
    """Create a spectrum based on the superquantile (or conditional value-at-risk) for ``n`` samples.
    
    Args:
      batch_size: the batch size.
      tail_prob: the proportion of largest elements to keep 
        in the loss computation, i.e. ``k/n`` for the top-k loss.
    
    Returns:
      spectrum
        a sorted vector of ``n`` weights on each training example.
    """
    if tail_prob < 0. or tail_prob > 1.:
        raise ValueError(
            "The proportion of largest elements to keep, tail_prob must be "
            "positive and less than 1. "
            f"Found 'tail_prob'={tail_prob}"
            )
    spectrum = np.zeros(batch_size, dtype=np.float64)
    idx = math.floor(batch_size * tail_prob)
    frac = 1 - (batch_size - idx - 1) / (batch_size * (1 - tail_prob))
    if frac > 1e-12:
        spectrum[idx] = frac
        spectrum[(idx + 1) :] = 1 / (batch_size * (1 - tail_prob))
    else:
        spectrum[idx:] = 1 / (batch_size - idx)
    return spectrum


def make_extremile_spectrum(batch_size: int, n_draws: float):
    """Create a spectrum based on the extremile for ``n`` samples.

    The spectrum is chosen so that the expectation of the loss vector 
    under this spectrum equals the uniform expected maximum of ``n_draws``
    elements from the loss vector. 

    See [Dauoia (2019)](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1498348) for more information.
    
    Args:
      batch_size: the batch size.
      n_draws: the number of independent draws from the
        loss vector. It can be fractional.
    
    Returns:
      spectrum
        a sorted vector of ``n`` weights on each training example.
    """
    if n_draws < 0.:
        raise ValueError(
            "The number of independent draws from the loss vector "
            "must be positive 0.. "
            f"Found 'n_draws'={n_draws}"
        )
    spectrum = (
        (np.arange(batch_size, dtype=np.float64) + 1) ** n_draws
        - np.arange(batch_size, dtype=np.float64) ** n_draws
    ) / (batch_size ** n_draws)
    return spectrum


def make_esrm_spectrum(batch_size: int, risk_param: float):
    """Create a spectrum based on the exponential spectral risk measure (ESRM) for ``n`` samples.

    See [Cotter (2006)](https://www.sciencedirect.com/science/article/pii/S0378426606001373) for more information.
    
    Args:
      batch_size: the batch size.
      risk_param: The ``R`` parameter from Cotter (2006).
    
    Returns:
      spectrum
        a sorted vector of ``n`` weights on each training example.
    """
    #TODO(ronakdm): add a check for the value of risk_param
    upper = np.exp(risk_param * ((np.arange(batch_size, dtype=np.float64) + 1) / batch_size))
    lower = np.exp(risk_param * (np.arange(batch_size, dtype=np.float64) / batch_size))
    return math.exp(-risk_param) * (upper - lower) / (1 - math.exp(-risk_param))