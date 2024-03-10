import math
import numpy as np

from functools import partial
import math
import torch
import numpy as np

from diropt.src.pav import l2_centered_isotonic_regression, neg_entropy_centered_isotonic_regression

def make_spectral_risk_measure(
        spectrum,
        penalty="chi2",
        shift_cost=0.0,
    ):
    return partial(spectral_risk_measure_maximization_oracle, spectrum, shift_cost, penalty)

def spectral_risk_measure_maximization_oracle(spectrum, shift_cost, penalty, losses):
    # TODO: Should this be a tolerance parameter?
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

def make_superquantile_spectrum(n, tail_prob):
    spectrum = np.zeros(n, dtype=np.float64)
    idx = math.floor(n * tail_prob)
    frac = 1 - (n - idx - 1) / (n * (1 - tail_prob))
    if frac > 1e-12:
        spectrum[idx] = frac
        spectrum[(idx + 1) :] = 1 / (n * (1 - tail_prob))
    else:
        spectrum[idx:] = 1 / (n - idx)
    return spectrum

def make_extremile_spectrum(n, n_draws):
    return (
        (np.arange(n, dtype=np.float64) + 1) ** n_draws
        - np.arange(n, dtype=np.float64) ** n_draws
    ) / (n ** n_draws)


def make_esrm_spectrum(n, risk_param):
    upper = np.exp(risk_param * ((np.arange(n, dtype=np.float64) + 1) / n))
    lower = np.exp(risk_param * (np.arange(n, dtype=np.float64) / n))
    return math.exp(-risk_param) * (upper - lower) / (1 - math.exp(-risk_param))