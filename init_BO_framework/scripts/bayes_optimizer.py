# bayes_optimizer.py

import numpy as np
import os
from .helpers import FE, FM, FD, make_surrogate, expected_improvement, UCB, thompson_sampling, gradient_opt, random_opt

class BayesianOptimizer:
    FUNC_MAP = {"FE": "FE", "FM": "FM", "FD": "FD"}  # only names here; actual call by passing your own function

    def __init__(
        self,
        func,                         # now pass the actual function, e.g. FE
        feature_columns: list,        # e.g. ['x1','x2',...]
        target_column: str = "y",     # e.g. "y"
        surrogate_kind: str = "gp",
        acq: str = "ei",
        xi: float = 0.01,
        kappa: float = 2.0,
        seed: int = None,
        opt_method: str = "grad_opt"
    ):
        self.func = func
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.dim = len(feature_columns)
        self.surrogate_kind = surrogate_kind
        self.acq = acq
        self.xi = xi
        self.kappa = kappa
        self.seed = seed
        self.surrogate = None
        self.opt_method = opt_method

    def optimize(self, x_init: np.ndarray, y_init: np.ndarray, bounds: list, n_iter: int):
        if self.seed is not None:
            np.random.seed(self.seed)
        x, y = x_init.copy(), y_init.copy()
        self.surrogate = make_surrogate(self.surrogate_kind)
        pred_y = []
        y_train_max = y_init.max()
        bounds_arr = np.array(bounds)

        for i in range(n_iter):
            self.surrogate.fit(x, y)
            y_best = y.max()
            if self.acq == "ei":
                acq_func = lambda xc, s: expected_improvement(xc, s, y_best, xi=self.xi)
            elif self.acq == "ucb":
                acq_func = lambda xc, s: UCB(xc, s, kappa=self.kappa)
            elif self.acq == "ts":
                acq_func = lambda xc, s: thompson_sampling(xc, s)
            else:
                raise ValueError(f"Unknown acquisition: {self.acq}")

            if self.opt_method == "grad_opt":
                x_next = gradient_opt(bounds_arr, self.surrogate, acq_func)
            elif self.opt_method == "rand_opt":
                x_next = random_opt(bounds_arr, self.surrogate, acq_func, n_samples=1000)
            else:
                raise ValueError(f"Unknown acquisition optimization: {self.opt_method}")
            y_next = self.func(x_next.reshape(1, -1))[0]
            x = np.vstack([x, x_next])
            y = np.append(y, y_next)
            pred_y.append(y_next)

        return x, y, pred_y, y_train_max