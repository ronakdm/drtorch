"""Functions for implementing the pool adjacent violators (PAV) algorithm using just-in-time compilation."""
import numpy as np
import numpy.typing as npt
from numba import jit

@jit(nopython=True)
def l2_centered_isotonic_regression(losses: npt.NDArray[np.float32], spectrum: npt.NDArray[np.float32]):
    """Solution to the isotonic regression problem when using the centered $\ell_2$ loss.
    
    Args:
      spectrum: a Numpy array containing the spectrum weights, which should be the same length as the batch size.
      losses: a Numpy array containing the loss on each example in the batch. These are the labels for isotonic regression.
    
    Returns:
      sample_weight
        a set of $n$ weights on each training example in the batch.
    """
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

    # Expand function so numba understands.
    sol = np.zeros((n,))
    i = 0
    for j in range(len(end_points)):
        end_point = end_points[j]
        sol[i : end_point + 1] = means[j]
        i = end_point + 1
    return sol



@jit(nopython=True)
def neg_entropy_centered_isotonic_regression(losses: npt.NDArray[np.float32], spectrum: npt.NDArray[np.float32]):
    """Solution to the isotonic regression problem when using the centered negative entropy loss.
    
    Args:
      spectrum: a Numpy array containing the spectrum weights, which should be the same length as the batch size.
      losses: a Numpy array containing the loss on each example in the batch. These are the labels for isotonic regression.
    
    Returns:
      sample_weight
        a set of $n$ weights on each training example in the batch.
    """
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
            # TODO: Find work around: the more numerically stable functions do not work with jit.
            # new_lse_loss = np.logaddexp(lse_losses[-1], prev_lse_loss)
            # new_lse_log_spectrum = np.logaddexp(lse_log_spectrum[-1], prev_lse_log_spectrum)
            new_lse_loss = np.log(np.exp(lse_losses[-1]) + np.exp(prev_lse_loss))
            new_lse_log_spectrum = np.log(np.exp(lse_log_spectrum[-1]) + np.exp(prev_lse_log_spectrum))
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