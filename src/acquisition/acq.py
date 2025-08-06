# src/acquisition/acq.py

from __future__ import annotations
import numpy as np
from scipy.stats import norm

def _predict(gp, X):
  """Always flatten the mean / std to 1-D (n,)."""
  mu, sigma = gp.predict(X, return_std=True)
  mu    = np.asarray(mu).reshape(-1)
  sigma = np.asarray(sigma).reshape(-1)
  return mu, sigma

def expected_improvement(X, gp, best_y: float, xi: float = 0.01):
  mu, sigma = _predict(gp, X)
  with np.errstate(divide="warn", invalid="warn"):
      Z  = (mu - best_y - xi) / sigma
      ei = (mu - best_y - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
      ei[sigma <= 1e-12] = 0.0                       # avoid shape mismatch
  return ei

def upper_confidence_bound(X, gp, kappa: float = 0.0):
  mu, sigma = _predict(gp, X)
  return mu + kappa * sigma

def thompson_sampling(X, gp):
  mu, sigma = _predict(gp, X)
  return np.random.normal(mu, sigma)


