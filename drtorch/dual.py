import numpy as np
from numba import jit
from functools import partial
import math
import torch

def make_spectral_risk_measure(
        n, 
        spectrum_type="superquantile", 
        superquantile_tail_probability=0.5,
        extremile_num_draws=2.0, 
        esrm_risk_aversion=1.0,
        shift_cost=0.0, 
        penalty="chi2"
    ):
    if spectrum_type == "superquantile":
        spectrum = compute_superquantile_spectrum(n, superquantile_tail_probability)
    elif spectrum_type == "extremile":
        spectrum = compute_extremile_spectrum(n, extremile_num_draws)
    elif spectrum_type == "esrm":
        spectrum = compute_esrm_spectrum(n, esrm_risk_aversion)
    return partial(spectral_risk_measure_maximization_oracle, spectrum, shift_cost, penalty)

def spectral_risk_measure_maximization_oracle(spectrum, shift_cost, penalty, losses):
    # TODO: Should this be a tolerance parameter?
    if shift_cost < 1e-16:
        print(shift_cost)
        print("lol")
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

@jit(nopython=True)
def l2_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    means = [losses[0] + 1 / n - spectrum[0]]
    counts = [1]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] + 1 / n - spectrum[i])
        counts.append(1)
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_count, prev_end_point = (
                means.pop(),
                counts.pop(),
                end_points.pop(),
            )
            means[-1] = (counts[-1] * means[-1] + prev_count * prev_mean) / (
                counts[-1] + prev_count
            )
            counts[-1] = counts[-1] + prev_count
            end_points[-1] = prev_end_point

    # Previous output without numba
    # sol = output_sol_iso_reg(end_points, means, n)

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol


# TODO: Find work around for jit with KL penalty.
# @jit(nopython=True)
def neg_entropy_centered_isotonic_regression(losses, spectrum):
    n = len(losses)
    logn = np.log(n)
    log_spectrum = np.log(spectrum)

    lse_losses = [losses[0]]
    lse_log_spectrum = [log_spectrum[0]]
    means = [losses[0] - log_spectrum[0] - logn]
    end_points = [0]
    for i in range(1, n):
        means.append(losses[i] - log_spectrum[i] - logn)
        lse_losses.append(losses[i])
        lse_log_spectrum.append(log_spectrum[i])
        end_points.append(i)
        while len(means) > 1 and means[-2] >= means[-1]:
            prev_mean, prev_lse_loss, prev_lse_log_spectrum, prev_end_point = (
                means.pop(),
                lse_losses.pop(),
                lse_log_spectrum.pop(),
                end_points.pop(),
            )
            new_lse_loss = np.logaddexp(lse_losses[-1], prev_lse_loss)
            new_lse_log_spectrum = np.logaddexp(lse_log_spectrum[-1], prev_lse_log_spectrum)
            means[-1] = new_lse_loss - new_lse_log_spectrum - logn
            lse_losses[-1], lse_log_spectrum[-1] = new_lse_loss, new_lse_log_spectrum
            end_points[-1] = prev_end_point

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol

def compute_superquantile_spectrum(n, theta):
    spectrum = np.zeros(n, dtype=np.float64)
    idx = math.floor(n * theta)
    frac = 1 - (n - idx - 1) / (n * (1 - theta))
    if frac > 1e-12:
        spectrum[idx] = frac
        spectrum[(idx + 1) :] = 1 / (n * (1 - theta))
    else:
        spectrum[idx:] = 1 / (n - idx)
    return spectrum

def compute_extremile_spectrum(n, r):
    return (
        (np.arange(n, dtype=np.float64) + 1) ** r
        - np.arange(n, dtype=np.float64) ** r
    ) / (n**r)


def compute_esrm_spectrum(n, rho):
    upper = np.exp(rho * ((np.arange(n, dtype=np.float64) + 1) / n))
    lower = np.exp(rho * (np.arange(n, dtype=np.float64) / n))
    return math.exp(-rho) * (upper - lower) / (1 - math.exp(-rho))