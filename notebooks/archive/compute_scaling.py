import os
import time
import json
import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskMultiFidelityGP
from botorch.fit import fit_gpytorch_mll
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm

# --- fidelity map ---
FIDELITY_MAP = {
    'y_low':    0.1,    # low fidelity
    'y_medium': 0.5,    # medium fidelity
    'y_high':   0.9,    # high fidelity
}

# 1) Load data
def load_json_to_df(path: str) -> pd.DataFrame:
    """Load newline-delimited JSON into a DataFrame."""
    with open(path, 'r') as f:
        records = [json.loads(line) for line in f]
    return pd.DataFrame(records)

# 2) Assign fidelities
def assign_fidelities(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From DataFrame with 'embedding' plus fidelity columns, build:
      X_rep: (N*K, D) repeated embeddings
      Y:     (N*K,) stacked fidelity values
      F:     (N*K,) stacked fidelity levels
    """
    X = np.vstack(df['embedding'].values).astype(np.float32)
    y_list, f_list = [], []
    for col, fid in FIDELITY_MAP.items():
        y_col = df[col].values.astype(np.float32).reshape(-1,1)
        f_col = np.full((df.shape[0],1), float(fid), dtype=np.float32)
        y_list.append(y_col)
        f_list.append(f_col)
    Y     = np.vstack(y_list).ravel()
    F     = np.vstack(f_list).ravel()
    X_rep = np.vstack([X for _ in FIDELITY_MAP])
    return X_rep, Y, F

# 3) Acquisition helpers
def _predict_with_uncertainty(model, x_cand: np.ndarray):
    x = torch.from_numpy(x_cand).float()
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post  = model.posterior(x)
        mu    = post.mean.cpu().numpy().ravel()
        sigma = post.variance.clamp_min(1e-9).sqrt().cpu().numpy().ravel()
    return mu, sigma

def expected_improvement(x_cand, model, y_best, xi=0.01):
    mu, sigma = _predict_with_uncertainty(model, x_cand)
    with np.errstate(divide='warn', invalid='warn'):
        Z  = (mu - y_best - xi) / sigma
        ei = (mu - y_best - xi)*norm.cdf(Z) + sigma*norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def gradient_opt(bounds, model, acq_func, n_restarts=10):
    best_x, best_val = None, -np.inf
    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:,0], bounds[:,1])
        res = minimize(
            lambda x: -acq_func(x.reshape(1,-1), model)[0],
            x0, bounds=bounds
        )
        val = -res.fun
        if val > best_val:
            best_x, best_val = res.x, val
    return best_x

# 4) Ground truth functions
def ground_truth(x: np.ndarray) -> np.ndarray:
    return -np.sum((x - 0.5)**2, axis=1) + np.sin(5 * np.sum(x, axis=1))

def ground_truth2(x: np.ndarray) -> np.ndarray:
    return np.sum(np.sqrt(x) * np.sin(x), axis=1)

# 5) Refactored multi-fidelity BO
def bayes_opt_mf(
    data_path: str,
    n_iter: int = 1,
    acq_func=expected_improvement,
    acq_opt=gradient_opt
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    1) loads JSON at `data_path`
    2) assigns X, Y, F
    3) fits initial multi-fidelity GP
    4) runs `n_iter` BO iterations
    returns updated X, Y, and history list of (x_next, y_next).
    """
    # a) read data
    df = load_json_to_df(data_path)
    X_init, Y_init, F_init = assign_fidelities(df)

    # b) get dims & bounds
    D       = X_init.shape[1]
    bounds  = np.array([[0.0, 1.0]] * D)
    fid_max = max(FIDELITY_MAP.values())

    # c) initial GP fit
    X_aug_init  = np.hstack([X_init, F_init.reshape(-1,1)])
    train_x     = torch.from_numpy(X_aug_init).double()
    train_y     = torch.from_numpy(Y_init).unsqueeze(-1).double()
    fidelity_col = train_x.shape[1] - 1

    surrogate = SingleTaskMultiFidelityGP(
        train_x, train_y, data_fidelities=[fidelity_col]
    )
    mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
    fit_gpytorch_mll(mll)

    # prepare for loop
    X_data, Y_data, F_data = X_init.copy(), Y_init.copy(), F_init.copy()
    history = []

    for it in tqdm(range(n_iter), desc=f"BO iter"):
        # refit GP on all data so far
        X_aug    = np.hstack([X_data, F_data.reshape(-1,1)])
        train_x = torch.from_numpy(X_aug).double()
        train_y = torch.from_numpy(Y_data).unsqueeze(-1).double()
        fidelity_col = train_x.shape[1] - 1

        surrogate = SingleTaskMultiFidelityGP(
            train_x, train_y, data_fidelities=[fidelity_col]
        )
        mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
        fit_gpytorch_mll(mll)

        # wrap acquisition to highest fidelity
        def wrapped_acq(x_emb, m=surrogate, y_max=Y_data.max()):
            fid_col = np.full((x_emb.shape[0],1), fid_max, dtype=np.float32)
            x_full  = np.hstack([x_emb, fid_col])
            return acq_func(x_full, m, y_max)

        # propose next
        x_next = acq_opt(bounds, surrogate, wrapped_acq)

        # evaluate with ground_truth2
        y_next = ground_truth2(x_next.reshape(1,-1))[0]

        # append
        X_data = np.vstack([X_data, x_next])
        Y_data = np.append(Y_data, y_next)
        F_data = np.append(F_data, fid_max)

        history.append((x_next, y_next))

    return X_data, Y_data, history

# 6) Batch-run BO over a directory with progress saving
def run_bo_on_dir(directory: str, processed_dir: str = '../data/processed', n_iter: int = 1, 
                  embed_dims: list[int] = None, n_samples_list: list[int] = None) -> pd.DataFrame:
    """
    Runs bayes_opt_mf in strict order over embed_dims and n_samples_list, using file names:
      synth_D{embed_dim}_N{n_samples}_ground_truth.json

    Records for each dataset:
      - dataset filename
      - embed_dim
      - n_samples
      - runtime in seconds

    Saves progress after each file to:
      {processed_dir}/bo_stats.csv (overwriting on each update)

    Returns the final DataFrame of results.
    """

    # default order if none provided
    if embed_dims is None:
        embed_dims = sorted([2, 4, 8, 16, 32, 64, 256, 512, 1024])
    if n_samples_list is None:
        n_samples_list = sorted([100, 1_000])

    os.makedirs(processed_dir, exist_ok=True)
    results = []

    # iterate in strict order
    for d in embed_dims:
        for n in n_samples_list:
            fname = f'synth_D{d}_N{n}_ground_truth.json'
            path = os.path.join(directory, fname)

            if not os.path.isfile(path):
                # skip missing files
                print(f"Warning: file not found, skipping: {fname}")
                continue

            # read minimal info
            df = load_json_to_df(path)
            embed_dim = len(df['embedding'].iloc[0])
            n_samples = df.shape[0]

            start = time.time()
            _, _, _ = bayes_opt_mf(path, n_iter=n_iter)
            runtime = time.time() - start

            results.append({
                'dataset':   fname,
                'embed_dim': embed_dim,
                'n_samples': n_samples,
                'runtime_s': runtime
            })

            # save progress after each dataset
            stats_df = pd.DataFrame(results)
            stats_df.to_csv(
                os.path.join(processed_dir, 'bo_stats.csv'),
                index=False
            )

    return stats_df

# --- example usage ---
if __name__ == '__main__':

    # batch run example
    stats_df = run_bo_on_dir(
        '../data/raw/synth_data',
        processed_dir='../data/processed',
        n_iter=1
    )
    print(stats_df)
