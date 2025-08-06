# src/utils/make_graphs.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from utils.utils import populate_training_metrics

# try optional UMAP
try:
   import umap
except ImportError:
   umap = None


def plot_batch_means(
   jsonl_paths: list[str],
   metric: str = "y_high",
   labels: list[str] | None = None,
   colors: list[str] | None = None,
   markers: list[str] | None = None,
   title: str | None = None,
   xlabel: str | None = None,
   ylabel: str | None = None,
   save_path: str | None = None
):
   """
   Plot per-iteration mean ± SEM for up to two runs,
   styled to match the attached Merck figure.
   """
   # defaults
   if labels   is None: labels   = [p.split("/")[-1] for p in jsonl_paths]
   if colors   is None: colors   = ["#ff7f0e", "#36827F"]     # orange, merck-teal
   if markers  is None: markers  = ["o", "s"]                # circle, square

   plt.figure(figsize=(6,4))
   for path, lbl, col, m in zip(jsonl_paths, labels, colors, markers):
       df = pd.read_json(path, lines=True)
       grp = df.groupby("iter")[metric]
       means = grp.mean()
       sems  = grp.sem()

       plt.errorbar(
           means.index, means.values, yerr=sems.values,
           marker=m,
           linestyle='-',
           color=col,
           linewidth=2,
           markersize=6,
           capsize=4,
           capthick=1.5,
           label=lbl
       )

   # grid + spines
   plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
   for spine in plt.gca().spines.values():
       spine.set_visible(True)

   plt.xlabel(xlabel or "Iteration", fontsize=11)
   plt.ylabel(ylabel or metric, fontsize=11)
   if title:
       plt.title(title, fontsize=12, pad=10)

   plt.legend(frameon=False, fontsize=10)
   plt.tight_layout()

   if save_path:
       plt.savefig(save_path, dpi=300)
   plt.show()


def plot_topology_3d(
   jsonl_path: str,
   reduce_method: str = "tsne",
   metric: str = "y_high",
   grid_size: int = 50,
   cmap_name: str = "Reds",
   elev: float = 30,
   azim: float = 120,
   connect_line_color: str = "k",
   connect_line_width: float = 1.0,
   save_path: str | None = None
):
   """
   3D fitness‐landscape surface + BO trajectory.

   - 2D reduction (TSNE/UMAP) of 1024-D embeddings.
   - GP fit on (x2,y2) → y_high; surface over dense grid.
   - White background with grid lines.
   - Training points with white fill, black border.
   - BO points colored by iteration via Reds cmap.
   - Path line from highest‐fitness training point through BO picks.
   """
   # 1) load & reduce (keep ALL rows for dimensionality reduction)
   df = pd.read_json(jsonl_path, lines=True)
   X = np.vstack(df["pca_embed"].values)
   if reduce_method.lower() == "umap" and umap:
       proj = umap.UMAP(n_components=2).fit_transform(X)
   else:
       proj = TSNE(n_components=2, perplexity=30).fit_transform(X)
   df["x2"], df["y2"] = proj[:,0], proj[:,1]

   # Filter for non-null metric values for GP fitting and plotting
   df_valid = df.dropna(subset=[metric])

   # 2) fit GP (only on valid metric observations)
   kernel = C(1.0, (1e-3,1e3)) * Matern(length_scale=1.0, nu=2.5)
   gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
   gp.fit(df_valid[["x2","y2"]], df_valid[metric])

   # 3) grid & predict (use full range from all points, but predict with GP trained on valid data)
   xi, xa = df["x2"].min(), df["x2"].max()
   yi, ya = df["y2"].min(), df["y2"].max()
   xs = np.linspace(xi, xa, grid_size)
   ys = np.linspace(yi, ya, grid_size)
   XX, YY = np.meshgrid(xs, ys)
   grid = np.vstack([XX.ravel(), YY.ravel()]).T
   Z = gp.predict(grid).reshape(XX.shape)

   # 4) plot setup
   fig = plt.figure(figsize=(10,7))
   ax  = fig.add_subplot(111, projection="3d")
   # white background panes + grid lines
   ax.set_facecolor('white')
   ax.xaxis.set_pane_color((1,1,1,1))
   ax.yaxis.set_pane_color((1,1,1,1))
   ax.zaxis.set_pane_color((1,1,1,1))
   ax.grid(True)

   # 5) surface
   ax.plot_surface(
       XX, YY, Z,
       rstride=1, cstride=1,
       cmap="viridis", alpha=0.6
   )

   # 6) training points (only plot those with valid metric values)
   tr = df_valid[df_valid["seq_id"] < 1024]
   ax.scatter(
       tr["x2"], tr["y2"], tr[metric],
       marker="o",
       facecolors="white",
       edgecolors="k",
       linewidths=1.0,
       s=50,
       label="Training Sequences"
   )

   # 7) BO-selected points (only plot those with valid metric values)
   bo = df_valid[(df_valid["seq_id"] >= 1024) & (df_valid["selected"])]
   iters = bo["iter"].values
   norm  = plt.Normalize(vmin=iters.min(), vmax=iters.max())
   cmap  = plt.cm.get_cmap(cmap_name)
   colors = cmap(norm(iters))
   ax.scatter(
       bo["x2"], bo["y2"], bo[metric],
       color=colors,
       edgecolors="k",
       s=50,
       label=None
   )

   # iteration colorbar
   mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
   mappable.set_array([])
   fig.colorbar(mappable, ax=ax, pad=0.1, label="Iteration")

   # find the single highest‐fitness training point
   start = tr.loc[tr[metric].idxmax()]

   # compute **per‐iteration** means of x2, y2, y_high
   batch_means = (
       bo.groupby("iter")
       .agg(x2=("x2","mean"),
           y2=("y2","mean"),
           metric_var=(metric,"mean"))
       .sort_index()
   )

   # build the line from start → mean₀ → mean₁ → …
   xs = [start.x2] + batch_means["x2"].tolist()
   ys = [start.y2] + batch_means["y2"].tolist()
   zs = [start[metric]] + batch_means["metric_var"].tolist()

   ax.plot(
       xs, ys, zs,
       color='k',
       linewidth=1.5,
       linestyle='--',
       label='Mean BO path'
   )

   ax.set_xlabel("Dim 1")
   ax.set_ylabel("Dim 2")
   ax.set_zlabel("In-Silico Fitness")
   ax.view_init(elev=elev, azim=azim)

   red_dot = Line2D(
       [0], [0],
       marker='o',
       color='w',
       markerfacecolor='maroon',
       markeredgecolor='k',
       markeredgewidth=0.5,   # ← match your training-point border
       markersize=8,
       label='BO-Selected Sequences'
   )

   handles, labels = ax.get_legend_handles_labels()
   handles.append(red_dot)
   labels.append('BO-Selected Sequences')

   # grab your existing handles (training & mean‐path) and append the red dot
   ax.legend(handles=handles, labels=labels, loc='upper left')

   plt.tight_layout()

   if save_path:
       plt.savefig(save_path)
   plt.show()