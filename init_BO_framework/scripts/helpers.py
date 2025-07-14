# helpers.py

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# --- Benchmark functions ---
def FE(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return -np.sum((x - 5)**2, axis=1) / d + np.exp(-np.sum(x**2, axis=1)) + 25


def FM(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return np.sum(x**4 - 16*(x**2) + 5*x, axis=1) / d


def FD(x: np.ndarray) -> np.ndarray:
    return np.sum(np.sqrt(x) * np.sin(x), axis=1)


def find_global_max(func, bounds: np.ndarray, n_restarts: int = 500):
    best_x, best_val = None, -np.inf
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        res = minimize(
            lambda x: -func(x.reshape(1, -1))[0],
            x0,
            bounds=bounds
        )
        val = -res.fun
        if val > best_val:
            best_val, best_x = val, res.x
    return best_x, best_val

# --- Surrogate factory ---
def make_surrogate(kind="gp", **kwargs):
    if kind == "gp":
        kernel = ConstantKernel(1.0) * Matern(nu=2.5)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, **kwargs)
    elif kind == "blr":
        return BayesianRidge(**kwargs)
    elif kind == "knn":
        return KNeighborsRegressor(n_neighbors=5, **kwargs)
    elif kind == "rf":
        return RandomForestRegressor(n_estimators=100, **kwargs)
    else:
        raise ValueError(f"Unknown surrogate: {kind}")

# --- Unified prediction with uncertainty ---
def _predict_with_uncertainty(surrogate, X: np.ndarray):
    # Random Forest: empirical tree-wise variance
    if isinstance(surrogate, RandomForestRegressor):
        preds = np.stack([tree.predict(X) for tree in surrogate.estimators_], axis=1)
        mu, sigma = preds.mean(axis=1), preds.std(axis=1)
    # Bayesian Ridge: analytic std
    elif isinstance(surrogate, BayesianRidge):
        mu, sigma = surrogate.predict(X, return_std=True)
    # KNN: sample std of neighbors' targets
    elif isinstance(surrogate, KNeighborsRegressor):
        idx = surrogate.kneighbors(X, return_distance=False)
        neigh_y = surrogate._y[idx]
        mu, sigma = neigh_y.mean(axis=1), neigh_y.std(axis=1)
    # Gaussian Process: return_std
    else:
        mu, sigma = surrogate.predict(X, return_std=True)
    return mu, sigma

# --- Acquisition functions ---
def expected_improvement(x_cand: np.ndarray, surrogate, y_best: float, xi: float = 0.01) -> np.ndarray:
    mu, sigma = _predict_with_uncertainty(surrogate, x_cand)
    with np.errstate(divide="warn", invalid="warn"):
        Z = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def UCB(x_cand: np.ndarray, surrogate, kappa: float = 2.0) -> np.ndarray:
    mu, sigma = _predict_with_uncertainty(surrogate, x_cand)
    return mu + kappa * sigma


def thompson_sampling(x_cand: np.ndarray, surrogate) -> np.ndarray:
    if isinstance(surrogate, GaussianProcessRegressor):
        mu, cov = surrogate.predict(x_cand, return_cov=True)
        return np.random.multivariate_normal(mu, cov)
    mu, sigma = _predict_with_uncertainty(surrogate, x_cand)
    return np.random.normal(mu, sigma)

# --- Acquisition optimization routines ---
def gradient_opt(bounds: np.ndarray, surrogate, acq_func, n_restarts: int = 10):
    best_x, best_val = None, -np.inf
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
        res = minimize(
            lambda x: -acq_func(x.reshape(1, -1), surrogate)[0],
            x0,
            bounds=bounds
        )
        val = -res.fun
        if val > best_val:
            best_x, best_val = res.x, val
    return best_x

def random_opt(bounds: np.ndarray, surrogate, acq_func, n_samples: int = 1000) -> np.ndarray:
    X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, bounds.shape[0]))
    vals = acq_func(X, surrogate)
    return X[int(np.argmax(vals))]