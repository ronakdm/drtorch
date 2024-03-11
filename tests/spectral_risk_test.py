"""Tests for Pool Adjacent Violator Algorithm."""

from absl.testing import absltest

import torch
import diropt._src.spectral_risk as spectral_risk


class SpectralRiskTest(absltest.TestCase):

  def test_spectral_risk(self):
    losses = ...
    weights = ...
    self.assertEqual(..., ...)


if __name__ == '__main__':
  absltest.main()