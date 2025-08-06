# src/models/gp_model.py

from __future__ import annotations
import torch, gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np


class DevelopabilityGP:
  """Tiny convenience wrapper around BoTorch's `SingleTaskGP`."""

  def __init__(self, train_x: np.ndarray, train_y: np.ndarray):
      self.update(train_x, train_y)

  # ---------------------------------------------------------------- public
  def predict(self, x: np.ndarray, return_std: bool = False):
      self._model.eval()
      with torch.no_grad(), gpytorch.settings.fast_pred_var():
          post = self._model.posterior(torch.as_tensor(x).double())
      mu = post.mean.squeeze(-1).cpu().numpy()
      if return_std:
          std = post.variance.clamp_min(1e-9).sqrt().cpu().numpy()
          return mu, std
      return mu

  def update(self, x: np.ndarray, y: np.ndarray):
      x_t = torch.as_tensor(x).double()
      y_t = torch.as_tensor(y).double().unsqueeze(-1)
      self._model = SingleTaskGP(x_t, y_t)
      fit_gpytorch_mll(ExactMarginalLogLikelihood(self._model.likelihood, self._model))
      return self