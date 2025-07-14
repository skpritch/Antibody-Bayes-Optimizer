# scripts/make_graphs.py

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

def plot_convergence(pred_y, y_train_max, global_max, model_name, out_dir=None):
    """
    Plot the convergence trajectory of Bayesian Optimization.
    """
    plt.figure()
    try:
        mean_y, sem_y = pred_y
        plt.errorbar(
            np.arange(len(mean_y)),
            mean_y,
            yerr=sem_y,
            fmt='-o',
            label="Mean BO ± SEM"
        )
    except Exception:
        plt.plot(pred_y, marker="o", label="BO Guesses")
    plt.axhline(y=y_train_max, color="navy", linestyle="--", label="Train max")
    plt.axhline(y=global_max, color="red", linestyle="-", label="Global max")
    plt.xlabel("Iteration")
    plt.ylabel("Output of Selected x_next")
    plt.title(f"{model_name} Convergence")
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    if out_dir:  # ← EDIT: only save if an out_dir is provided
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"convergence_{model_name}.png"))
        plt.close(fig)
    return fig  # ← EDIT: return the Matplotlib Figure for front-end display

def plot_suggestions(df, feature, global_max, best_x, best_y, target_column, func_name, dim, out_dir=None):
    """
    Plot initial data and best suggestions from BO.
    """
    plt.figure()
    plt.scatter(df[feature], df[target_column], alpha=0.5, label="Initial data")
    n = len(best_x)
    cmap = cm.get_cmap("Reds")
    idxs = np.linspace(0, 1, n) if n > 1 else [1.0]
    colors = cmap(idxs)
    plt.scatter(best_x, best_y, c=colors, label="Suggested (darker → later)")
    plt.axhline(y=global_max, color="red", linestyle="-", label="Global max")
    plt.xlabel(feature)
    plt.ylabel(target_column)
    plt.title(f"{func_name} d={dim}")
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    if out_dir:  # ← EDIT: only save if out_dir is not None
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"scatter_{func_name}_d{dim}.png"))
        plt.close(fig)
    return fig  # ← EDIT

def plot_dataset(df, feature, global_max, target, func_name, dim, out_dir=None):
    """
    Plot the full dataset for a given function and dimension.
    """
    plt.figure()
    plt.scatter(df[feature], df[target], alpha=0.5, label="Initial data")
    plt.axhline(y=global_max, color="red", linestyle="-", label="Global max")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f"Full data: {func_name} d={dim}")
    plt.tight_layout()
    fig = plt.gcf()
    if out_dir:  # ← EDIT
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"dataset_{func_name}_d{dim}.png"))
        plt.close(fig)
    return fig  # ← EDIT
