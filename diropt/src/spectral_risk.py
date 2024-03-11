"""Functions relating to spectral risk measure ambiguity sets."""
import math
import numpy as np

from functools import partial
import math
import torch
import numpy as np
import numpy.typing as npt

from diropt.src.pav import l2_centered_isotonic_regression, neg_entropy_centered_isotonic_regression

def make_spectral_risk_measure(
        spectrum: npt.NDArray[np.float32],
        penalty: str="chi2",
        shift_cost: float=0.0,
    ):
    """Create a function which computes the sample weights from a vector of losses when using a spectral risk measure ambiguity set.
    
    Args:
      spectrum: a Numpy array containing the spectrum weights, which should be the same length as the batch size.
      penalty: either 'chi2' or 'kl' indicating which f-divergence to use as the dual regularizer.
      shift_cost: ehe non-negative dual regularization parameter.
    
    Returns:
      compute_sample_weight
        a function that maps $n$ losses to a vector of $n$ weights on each training example.
    """
    return partial(spectral_risk_measure_maximization_oracle, spectrum, shift_cost, penalty)

def spectral_risk_measure_maximization_oracle(
        spectrum: npt.NDArray[np.float32], 
        shift_cost: float,
        penalty: str,
        losses: npt.NDArray[np.float32]
    ):
    """Maximization oracle to compute the sample weights based on a particular spectral risk measure objective.
    
    Args:
      spectrum: a Numpy array containing the spectrum weights, which should be the same length as the batch size.
      shift_cost: ehe non-negative dual regularization parameter.
      penalty: either 'chi2' or 'kl' indicating which f-divergence to use as the dual regularizer.
      losses: a Numpy array containing the loss incurred by the model on each example in the batch.
    
    Returns:
      sample_weight
        a vector of $n$ weights on each training example.
    """
    if shift_cost < 1e-16:
        return torch.from_numpy(spectrum[np.argsort(np.argsort(losses))])
    n = len(losses)
    scaled_losses = losses / shift_cost
    perm = np.argsort(losses)
    sorted_losses = scaled_losses[perm]

    if penalty == "chi2":
        primal_sol = l2_centered_isotonic_regression(
            sorted_losses, spectrum
        )
    elif penalty == "kl":
        primal_sol = neg_entropy_centered_isotonic_regression(sorted_losses, spectrum)
    else:
        raise NotImplementedError
    inv_perm = np.argsort(perm)
    primal_sol = primal_sol[inv_perm]
    if penalty == "chi2":
        q = scaled_losses - primal_sol + 1 / n
    elif penalty == "kl":
        q = np.exp(scaled_losses - primal_sol) / n
    else:
        raise NotImplementedError
    return torch.from_numpy(q).float()

def make_superquantile_spectrum(n: int, tail_prob: float):
    """Create the spectrum (collection of $n$ sample weights) corresponding to the superquantile (or conditional value-at-risk).
    
    Args:
      n: the batch size.
      tail_prob: the proportion of largest elements to keep in the loss computation, i.e. $k/n$ for the top-$k$ loss.
    
    Returns:
      spectrum
        a sorted vector of $n$ weights on each training example.
    """
    spectrum = np.zeros(n, dtype=np.float64)
    idx = math.floor(n * tail_prob)
    frac = 1 - (n - idx - 1) / (n * (1 - tail_prob))
    if frac > 1e-12:
        spectrum[idx] = frac
        spectrum[(idx + 1) :] = 1 / (n * (1 - tail_prob))
    else:
        spectrum[idx:] = 1 / (n - idx)
    return spectrum

def make_extremile_spectrum(n: int, n_draws: float):
    """Create the spectrum (collection of $n$ sample weights) corresponding to the extremile. 
    The spectrum is chosen so that the expectation of the loss vector under this spectrum equals
    the uniform expected maximum of `n_draws` elements from the loss vector. 

    See [Dauoia (2019)](https://www.tandfonline.com/doi/full/10.1080/01621459.2018.1498348) for more information.
    
    Args:
      n: the batch size.
      n_draws: the number of independent draws from the loss vector to make the equality above true. It can be fractional.
    
    Returns:
      spectrum
        a sorted vector of $n$ weights on each training example.
    """
    return (
        (np.arange(n, dtype=np.float64) + 1) ** n_draws
        - np.arange(n, dtype=np.float64) ** n_draws
    ) / (n ** n_draws)


def make_esrm_spectrum(n: int, risk_param: float):
    """Create the spectrum (collection of $n$ sample weights) corresponding to the exponential spectral risk measure (ESRM). 
    See [Cotter (2006)](https://www.sciencedirect.com/science/article/pii/S0378426606001373) for more information.
    
    Args:
      n: the batch size.
      risk_param: The $R$ parameter from Cotter (2006).
    
    Returns:
      spectrum
        a sorted vector of $n$ weights on each training example.
    """
    upper = np.exp(risk_param * ((np.arange(n, dtype=np.float64) + 1) / n))
    lower = np.exp(risk_param * (np.arange(n, dtype=np.float64) / n))
    return math.exp(-risk_param) * (upper - lower) / (1 - math.exp(-risk_param))