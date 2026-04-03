import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple


sns.set_theme(style="whitegrid", context="talk")
sns.set_palette("tab10")

path = "ckpts/paper/clevrtex"
model_names = ["test0", "sigmoid_resunet_bw3_gr_unnorm"]
# model_names = ["test0", "sigmoid_resunet_bw3_gr_ws2"]
# model_names = ["test0", "sigmoid_resunet_bw3_gr_ws3"]
# model_names = ["test0", "sigmoid_resunet_bw3_gr_ws4"]

# model_names = ["sigmoid_resunet_bw3_gr", "sigmoid_resunet_bw5_gr"]
# model_names = ["test0", "sigmoid_resunet_bw5_gr_unnorm"]

# model_names = ["test0", "SIGMOID_resunet_bw3_gr"]
model_names = ["test0", "SIGMOID_resunet_bw3_gr2"]
# model_names = ["test0", "SIGMOID_resunet_bw5_gr"]
# model_names = ["test0_unnorm", "SIGMOID_resunet_bw5_gr_unnorm"]
# model_names = ["test0", "SIGMOID_resunet_bw3_gr_unnorm"]

metrics = {}

for folder in os.listdir(path):
    model = folder[2:]
    if model not in metrics:
        metrics[model] = []

    df = pd.read_csv(os.path.join(path, folder, "metrics.csv"))
    metrics[model].append({
        "nimgs": df.nimgs.values,
        "LPIPS": df.lpips.values,
        "FG-ARI": df.fari.values,
        "mIoU": df.miou.values,
        "mBO": df.mbo.values,
    })

def smooth_curve(y, window=9, poly=3):
    n = len(y)
    if n < 3:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

cmap_name = "coolwarm_r"
model_styles = {
    model_names[0]: "-",
    model_names[1]: "--",
    # "sigmoid_resunet_bw5_gr": ":",
}

fig, axes = plt.subplots(
    nrows=1,
    ncols=8,
    figsize=(20, 5),
    constrained_layout=True,
    gridspec_kw={
        "width_ratios": [1, 0.06, 0.05, 1, 0.06, 0.05, 1, 0.06],
    },
)

axes[2].set_axis_off()
axes[5].set_axis_off()

p_marker = 0

for i, (metric, lb, ub, ymin, ymax, xy, xytext) in enumerate([
    ("FG-ARI", 0.6, 0.8, 0.0, 0.9, (0.28, 0.85), (0.5, 0.55)),
    ("mIoU", 0.5, 0.65, 0.0, 0.7, (0.28, 0.6), (0.5, 0.38)),
    ("mBO", 0.5, 0.7, 0.0, 0.75, (0.28, 0.6), (0.5, 0.38))
]):
    ax = axes[i * 3]
    cax = axes[i * 3 + 1]

    norm = plt.Normalize(vmin=lb, vmax=ub)
    cmap = plt.get_cmap(cmap_name)

    for model in model_names:
        linestyle = model_styles.get(model, "-")

        for vals in metrics[model]:
            # x_smooth = smooth_curve(vals["LPIPS"])
            # y_smooth = smooth_curve(vals[metric])
            x_smooth = vals["LPIPS"]
            y_smooth = vals[metric]

            ax.plot(
                x_smooth,
                y_smooth,
                c=cmap(norm(vals[metric][-1])),
                linestyle=linestyle,
                linewidth=2,
                alpha=0.95,
            )

            ax.scatter(
                x_smooth[0],
                y_smooth[0],
                c="mediumseagreen",
                marker="o",
                s=50,
                edgecolor="black",
                linewidth=2,
                zorder=50,
            )

            ax.scatter(
                x_smooth[p_marker],
                y_smooth[p_marker],
                marker="P",
                s=80,
                c="mediumseagreen",
                edgecolor="black",
                linewidths=1,
                zorder=50,
            )

            ax.scatter(
                x_smooth[-1],
                y_smooth[-1],
                c="mediumseagreen",
                marker="^",
                s=50,
                edgecolor="black",
                linewidth=2,
                zorder=50,
            )

    ax.set_xlim(0.1, 0.8)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Reconstruction (LPIPS)", fontsize=15)
    ax.set_ylabel(f"Disentanglement ({metric})", fontsize=15)
    ax.tick_params(axis="y", left=True, labelsize=15)
    ax.tick_params(axis="x", bottom=True, labelsize=15)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"Final {metric}", fontsize=15)
    cbar.ax.tick_params(labelsize=15)

legend_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=2, label=model_names[0]),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label=model_names[1]),
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor="mediumseagreen", markeredgecolor="black",
           markeredgewidth=1, markersize=9, label="Start (0 imgs)"),
    Line2D([0], [0], marker="P", linestyle="None",
           markerfacecolor="mediumseagreen", markeredgecolor="black",
           markeredgewidth=1, markersize=10,
           label=f"Ref. ({metrics[model_names[0]][0]['nimgs'][p_marker] // 1000}K imgs)"),
    Line2D([0], [0], marker="^", linestyle="None",
           markerfacecolor="mediumseagreen", markeredgecolor="black",
           markeredgewidth=1, markersize=9,
           label=f"End ({metrics[model_names[0]][0]['nimgs'][-1] // 1_000_000}M imgs)"),
]

axes[-2].legend(
    handles=legend_handles,
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    facecolor="white",
    edgecolor="0.8",
    fontsize=15,
)

save_name = "_vs_".join(model_names)
plt.savefig(f"analysis_plots/ocl_analysis_traj_{save_name}.pdf")
plt.show()