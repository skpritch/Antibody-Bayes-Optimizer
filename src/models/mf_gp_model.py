# src/model/mg_gp_model.py

from __future__ import annotations
import numpy as np, torch, gpytorch
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Sequence, Dict


class MultiFidelityGP:
  def __init__(
      self,
      X: np.ndarray,
      y: np.ndarray,
      fidelities: np.ndarray,
      fidelity_values: Sequence[float] | Dict[str, float],
  ):
      if isinstance(fidelity_values, dict):
          # keep user‑specified order (key order must match Y columns)
          self._fid_values = np.asarray(list(fidelity_values.values()))
      else:
          self._fid_values = np.asarray(list(fidelity_values))
      self.fid_max = self._fid_values.max()
      self.update(X, y, fidelities)

  # -------------------------------------------------------------- predict
  def predict(self, X: np.ndarray, fidelity: float | None = None, return_std=False):
      fid = self.fid_max if fidelity is None else float(fidelity)
      X_aug = np.hstack([X, np.full((len(X), 1), fid, dtype=np.float32)])

      self._model.eval()
      with torch.no_grad(), gpytorch.settings.fast_pred_var():
          post = self._model.posterior(torch.as_tensor(X_aug).double())
      mu = post.mean.squeeze(-1).cpu().numpy()
      if return_std:
          std = post.variance.clamp_min(1e-9).sqrt().cpu().numpy()
          return mu, std
      return mu

  # --------------------------------------------------------------- update
  def update(self, X: np.ndarray, y: np.ndarray, fidelity: np.ndarray):
      X_aug = np.hstack([X, fidelity.reshape(-1, 1)])
      tx = torch.as_tensor(X_aug).double()
      ty = torch.as_tensor(y).double().unsqueeze(-1)

      fid_col = tx.shape[1] - 1
      self._model = SingleTaskMultiFidelityGP(tx, ty, data_fidelities=[fid_col])
      fit_gpytorch_mll(ExactMarginalLogLikelihood(self._model.likelihood, self._model))
      self.X, self.y, self.fidelity = X, y, fidelity
      return self

  def add_data(self, x_new: np.ndarray, y_new: float, fidelity: float | None = None):
      fid = self.fid_max if fidelity is None else float(fidelity)
      self.X = np.vstack([self.X, x_new])
      self.y = np.append(self.y, y_new)
      self.fidelity = np.append(self.fidelity, fid)
      return self.update(self.X, self.y, self.fidelity)