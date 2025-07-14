# run_experiments.py

import os
import logging
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from scripts.bayes_optimizer import BayesianOptimizer
from scripts.helpers import FE, FM, FD, find_global_max
from scripts.utils import load_data, preprocess
from scripts.make_graphs import plot_convergence, plot_dataset, plot_suggestions

import warnings
warnings.filterwarnings("ignore")

# CONFIG
FEATURE_SETS = {
    2: ["x1", "x2"],
    5: [f"x{i+1}" for i in range(5)],
    10: [f"x{i+1}" for i in range(10)],
    20: [f"x{i+1}" for i in range(20)],
}
TARGET_COLUMN = "y"
FUNCS = {"FE": FE, "FM": FM, "FD": FD}
N_INIT = 50        # initial points
N_ITER = 30
N_EXPS = 50
DATA_DIR = "datasets/synth_data/"
RESULT_DIR = "results/synth_results/"

logging.basicConfig(level=logging.INFO)

def run_experiments():
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_results = []

    for func_name, func in FUNCS.items():
        for dim, features in FEATURE_SETS.items():
            path = os.path.join(DATA_DIR, f"{func_name}_d{dim}_N{N_INIT}.json")
            x_raw, y, df = load_data(path, features, TARGET_COLUMN)
            x_init, _ = preprocess(x_raw)
            bounds = [[0, 10]] * dim

            # true maxima
            bounds_arr = np.array(bounds)
            _, global_max = find_global_max(func, bounds_arr)

            successes = 0
            best_x1_list, best_y_list = [], []
            all_preds = []    # will hold each trial’s pred_y sequence

            for seed in tqdm(range(N_EXPS), desc=f"{func_name}, d={dim}"):
                bo = BayesianOptimizer(
                    func=func,
                    feature_columns=features,
                    target_column=TARGET_COLUMN,
                    seed=seed
                )
                x_all, y_all, pred_y, y_train_max = bo.optimize(x_init, y, bounds, N_ITER)
                all_preds.append(pred_y)

                # pick best suggestion
                best_idx = int(np.argmax(pred_y))
                x_sugg = x_all[-N_ITER:][best_idx]
                y_sugg = pred_y[best_idx]

                best_x1_list.append(x_sugg[0])
                best_y_list.append(y_sugg)
                successes += int(y_sugg > y_train_max)

            prop = successes / N_EXPS
            mean_guess = np.mean(best_y_list)
            std_guess  = np.std(best_y_list)
            all_results.append({
                "Function":    func_name,
                "Dimension":   dim,
                "TrainMax":    y_train_max,
                "GlobalMax":   global_max,
                "MeanGuess":   mean_guess,
                "StdGuess":    std_guess,
                "SuccessRate": prop
            })

            # plots
            # 1) full dataset
            plot_dataset(df, features[0], global_max, TARGET_COLUMN, func_name, dim, RESULT_DIR)
            # 2) all suggestions overlaid
            plot_suggestions(df, features[0], global_max, best_x1_list, best_y_list, TARGET_COLUMN, func_name, dim, RESULT_DIR)
            # 3) mean ± SEM convergence
            preds_arr   = np.array(all_preds)            # shape (N_EXPS, N_ITER)
            mean_preds  = preds_arr.mean(axis=0)
            sem_preds   = preds_arr.std(axis=0) / np.sqrt(N_EXPS)
            plot_convergence(
                (mean_preds, sem_preds),
                y_train_max,
                global_max,
                f"{func_name}_d{dim}_avg",
                RESULT_DIR
            )

            # write a temporary CSV (overwritten each batch)
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(os.path.join(RESULT_DIR, "temp_results.csv"), index=False)

    all_results.to_csv(os.path.join(RESULT_DIR, "final_results.csv"), index=False)
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    df = run_experiments()
    print(df)