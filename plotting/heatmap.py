import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech"
nimgs_target = 2944000   # training point of interest

metrics = ["dci", "factorvae", "mig", "mse", "lpips"]
recon_metrics = {"mse", "lpips"}   # lower is better → use reversed colormap

os.makedirs("analysis_plots", exist_ok=True)

# regex to grab model_seed and timestep_seed from names like "23v_cosine_edm-3"
name_re = re.compile(r"^(\d)(\d)v_")

rows = {}
model_seeds = set()
time_seeds = set()

for run_dir in os.listdir(base_dir):
    full_dir = os.path.join(base_dir, run_dir)
    if not os.path.isdir(full_dir):
        continue

    m = name_re.match(run_dir)
    if m is None:
        continue

    model_seed = int(m.group(1))
    time_seed = int(m.group(2))

    metrics_path = os.path.join(full_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        continue

    df = pd.read_csv(metrics_path)

    if "nimgs" not in df.columns:
        continue

    # choose exact or nearest nimgs
    if (df["nimgs"] == nimgs_target).any():
        row = df.loc[df["nimgs"] == nimgs_target].iloc[0]
    else:
        idx = (df["nimgs"] - nimgs_target).abs().argmin()
        row = df.iloc[idx]

    key = (model_seed, time_seed)
    rows[key] = row
    model_seeds.add(model_seed)
    time_seeds.add(time_seed)

# sort seeds
model_seeds = sorted(model_seeds)
time_seeds = sorted(time_seeds)

# build matrices
mats = {m: np.full((len(model_seeds), len(time_seeds)), np.nan, dtype=float)
        for m in metrics}

for (model_seed, time_seed), row in rows.items():
    i = model_seeds.index(model_seed)
    j = time_seeds.index(time_seed)
    for m in metrics:
        if m in row.index:
            mats[m][i, j] = row[m]

# ----- plotting panel -----

fig, axes = plt.subplots(
    1, len(metrics),
    figsize=(4 * len(metrics), 4),
    sharex=True,
    sharey=True
)

if len(metrics) == 1:
    axes = [axes]

for ax, m in zip(axes, metrics):

    mat = mats[m]

    # reversed colormap for recon metrics
    cmap = "viridis_r" if m in recon_metrics else "viridis"

    sns.heatmap(
        mat,
        xticklabels=time_seeds,
        yticklabels=model_seeds,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        ax=ax,
        cbar=False,
    )
    ax.set_title(m)
    ax.set_xlabel("timestep seed")
    ax.set_ylabel("model seed")

fig.suptitle(f"Metrics at {nimgs_target:,} nimgs", y=1.02)
plt.tight_layout()
out_path = f"analysis_plots/all_metrics_{nimgs_target}_heatmaps.pdf"
plt.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved {out_path}")
