"""Tests for Pool Adjacent Violator Algorithm."""

from absl.testing import absltest

import numpy as np

import sys
sys.path.append(".")
import diropt._src.spectral_risk as spectral_risk
import diropt._src.spectra as spectra


class SpectralRiskTest(absltest.TestCase):

  def test_spectral_risk(self):
    test_func = spectral_risk.spectral_risk_measure_maximization_oracle
    batch_size = 30
    uniform = [1. / batch_size for _ in range(batch_size)]

    for penalty in ["chi2", "kl"]:

      losses = np.ones(batch_size)
      spectrum = spectra.make_extremile_spectrum(batch_size, 2.0)
      weights = [elem.item() for elem in test_func(spectrum, 1.0, penalty, losses)]
      self.assertSequenceAlmostEqual(weights, uniform, places=6)

      np.random.seed(123)
      losses = np.sort(np.random.normal(size=(batch_size,)))
      spectrum = spectra.make_superquantile_spectrum(batch_size, 0.5)
      weights1 = test_func(spectrum, 1.0, penalty, losses)
      weights2 = test_func(spectrum, 2.0, penalty, losses)
      self.assertGreaterEqual(weights1[-1], weights2[-1])
      self.assertGreaterEqual(weights1.min().item(), -1e-12)
      self.assertAlmostEqual(weights1.sum().item(), 1., places=6)

      np.random.seed(456)
      losses = np.random.normal(size=(batch_size,))
      spectrum = spectra.make_esrm_spectrum(batch_size, 1.0)
      weights = [elem.item() for elem in test_func(spectrum, 0.1, penalty, losses)]
      self.assertTrue(np.all(np.argsort(weights) == np.argsort(losses)))



if __name__ == '__main__':
  absltest.main()