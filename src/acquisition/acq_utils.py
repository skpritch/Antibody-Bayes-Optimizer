# src/acquisition/acq_utils.py

from __future__ import annotations
from acquisition.acq import expected_improvement, upper_confidence_bound, thompson_sampling

def make_acq(acq: str, gp, y_best: float, *, xi=0.01, kappa=2.0):
  """
  Parameters
  ----------
  acq : "ei" | "ucb" | "ts"
  gp  : GP wrapper with .predict()
  y_best : current best objective value
  Returns
  -------
  callable: f(X_emb) -> np.ndarray
  """
  acq = acq.lower()
  if acq == "ei":
      return lambda X: expected_improvement(X, gp, y_best, xi)
  if acq == "ucb":
      return lambda X: upper_confidence_bound(X, gp, kappa)
  if acq == "ts":
      return lambda X: thompson_sampling(X, gp)
  raise ValueError(f"Unknown acquisition '{acq}'")
