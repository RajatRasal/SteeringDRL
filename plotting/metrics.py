# metrics.py
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
# ROOT = "ckpts/vdmpp/shapes3d_low_discrep..."
# ROOT = "ckpts/vdmpp/shapes3d_early_stage/..."
# ROOT = "ckpts/vdmpp_deep_longer_stable/shapes3d/"
# ROOT = "ckpts/vdmpp_deep_longer_stable2/shapes3d/"
# ROOT = "ckpts/vdmpp_deep_longer/shapes3d/"

# ROOT = "ckpts/vdmpp_deep_longer/shapes3d/"
# ROOT = "ckpts/vdmpp_deep_longer_stable/shapes3d"
# ROOT = "ckpts/vdmpp_deep_longer_stable/shapes3d"
# ROOT = "ckpts/vdmpp_deep/shapes3d"
# ROOT = "ckpts/vdmpp_deep_longer_stable2_improved/shapes3d/"
# ROOT = "ckpts/ocl-test"
# ROOT = "ckpts/final/voc"
# ROOT = "ckpts/final/coco"
# ROOT = "ckpts/final/shapes3d"
# ROOT = "ckpts/final/cars3d"
# ROOT = "ckpts/final/mpi3d"

# ROOT = "ckpts/final2/shapes3d"
# ROOT = "ckpts/final2/mpi3d"
# ROOT = "ckpts/final2/cars3d"

# ROOT = "ckpts/paper/shapes3d"
# ROOT = "ckpts/paper/cars3d"
ROOT = "ckpts/paper/mpi3d"

METRICS = ["dci", "factorvae", "mig", "lpips"]
# METRICS = ["ari", "fari", "miou", "mbo", "lpips"] # "mse", "lpips"]
# METRICS = ["fari", "miou", "mbo", "lpips"] # "mse", "lpips"]
# METRICS = ["ari", "fari", "miou", "mbo", "smbo", "lpips"] # "mse", "lpips"]

SAVE_FIGS = True
OUTDIR = "analysis_plots"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------
# helpers
# ----------------------------------------------------
def final_valid_value(df: pd.DataFrame, metric: str):
    """Return final valid (non-NaN) value from df[metric], else None."""
    if df is None or df.empty or metric not in df.columns:
        return None
    s = df[metric].dropna()
    if len(s) == 0:
        return None
    v = s.iloc[-1]
    try:
        return float(v)
    except Exception:
        return None


# ----------------------------------------------------
# 1. Walk the directory and gather metrics per model_setting
#    Folder naming convention:
#      "<seed><model_setting>"
#    e.g. "0shapes3d_v_cosine_sech"
# ----------------------------------------------------
results = defaultdict(list)  # final values per run
curves = defaultdict(list)   # full curves per run (nimgs + metrics)

for folder in os.listdir(ROOT):
    folder_path = os.path.join(ROOT, folder)
    # if not os.path.isdir(folder_path) or "skip_res1" not in folder_path:
    if not os.path.isdir(folder_path): #  or "skip_res1" not in folder_path:
        continue

    # expect folder names like "0shapes3d_v_cosine_sech"
    try:
        # seed = int(folder[0])
        # model_setting = int(folder[-1])
        seed = int(folder[:2])
        model_setting = folder[2:]
        # seed = int(folder[:1])
        # model_setting = folder[1:]
    except Exception:
        # skip folders not matching the pattern
        continue

    # if "sech" in model_setting:
    #     continue

    metrics_path = os.path.join(folder_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"Skipping {folder}, no metrics.csv")
        continue

    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        print(f"Skipping {folder}, failed to read metrics.csv: {e}")
        continue

    if ROOT.endswith("voc") or ROOT.endswith("coco"):
        df.reset_index(inplace=True)
        df.columns = ["nimgs","time",'loss','mse','lpips','ari','fari','miou','fmiou','mbo','sari','sfari','smiou','sfmiou','smbo','importance_sim']

    # store final values robustly (handles empty cols / NaNs)
    entry = {}
    for m in METRICS:
        entry[m] = final_valid_value(df, m)

    # optionally skip runs with no usable values
    if all(entry[m] is None for m in METRICS):
        print(f"Skipping {folder}, metrics.csv has no usable metric values")
        continue

    results[model_setting].append(entry)

    # store full curve for training-efficiency plot
    needed_cols = ["nimgs"] + [m for m in METRICS if m in df.columns]
    if "nimgs" in df.columns:
        curves[model_setting].append(df[needed_cols].copy())


# ----------------------------------------------------
# 2. Single figure: VIOLIN plots (1 row, N metrics)
#    Locatello-ish styling:
#      - each model has its own colour
#      - inner box/band
# ----------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")

valid_metrics = []
metric_data_list = []

for metric in METRICS:
    data = {}
    for model_setting, entries in results.items():
        vals = [e[metric] for e in entries if e[metric] is not None]
        if len(vals) > 0:
            data[model_setting] = vals

    if len(data) == 0:
        print(f"No data for {metric}")
        continue

    valid_metrics.append(metric)
    metric_data_list.append(data)

n_metrics = len(valid_metrics)
if n_metrics == 0:
    raise RuntimeError("No metrics found to plot. Check ROOT / metrics.csv columns.")

# stable model order across all plots
model_order = sorted(
    {ms for data in metric_data_list for ms in data.keys()}
)

# consistent palette: one colour per model
palette = dict(zip(model_order, sns.color_palette(n_colors=len(model_order))))

fig, axes = plt.subplots(1, n_metrics, figsize=(4.6 * n_metrics, 15), squeeze=False)
axes = axes[0]

for idx, (metric, data) in enumerate(zip(valid_metrics, metric_data_list)):
    ax = axes[idx]

    # long-form df for seaborn
    rows = []
    for ms, vals in data.items():
        for v in vals:
            rows.append({"model": ms, "value": float(v)})
    df_plot = pd.DataFrame(rows)

    sns.violinplot(
        data=df_plot,
        x="model",
        y="value",
        order=model_order,
        hue="model",              # colour by model
        hue_order=model_order,
        palette=palette,
        dodge=False,              # don't split / offset
        inner="box",              # inner band/box
        cut=0,                    # don't extend beyond min/max
        linewidth=1.1,
        saturation=1.0,
        ax=ax,
        legend=False,
    )

    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90)
    if idx == 0:
        ax.set_ylabel("Score")
    else:
        ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(f"{OUTDIR}/all_metrics_violins_row.png", dpi=300, bbox_inches="tight")
plt.show()


# ----------------------------------------------------
# 3. Mean + Std Error Bar PLOTS (1 row × N metrics) — no connecting lines
# ----------------------------------------------------
fig2, axes2 = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6), squeeze=False)
axes2 = axes2[0]

for idx, (metric, data) in enumerate(zip(valid_metrics, metric_data_list)):
    ax = axes2[idx]

    labels = list(data.keys())
    means = [pd.Series(v).mean() for v in data.values()]
    stds  = [pd.Series(v).std()  for v in data.values()]

    ax.errorbar(
        x=range(len(labels)),
        y=means,
        yerr=stds,
        fmt="o",          # dots only
        capsize=4,
    )
    ax.set_title(metric)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    if idx == 0:
        ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIGS:
    fig2.savefig(f"{OUTDIR}/all_metrics_mean_std_row.png", dpi=200, bbox_inches="tight")
plt.show()


# ----------------------------------------------------
# 4. Training efficiency vs nimgs (1 row × N metrics)
#    Uses groupby over nimgs to handle different run lengths
# ----------------------------------------------------
fig3, axes3 = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6), squeeze=False)
axes3 = axes3[0]

for idx, metric in enumerate(valid_metrics):
    ax = axes3[idx]

    for model_setting, runs in curves.items():
        # Only keep runs where this metric is present
        metric_runs = [
            r[["nimgs", metric]].dropna(subset=[metric])
            for r in runs
            if metric in r.columns and "nimgs" in r.columns
        ]
        if not metric_runs:
            continue

        # concat and aggregate
        concat = pd.concat(metric_runs, axis=0, ignore_index=True)
        grouped = concat.groupby("nimgs")[metric].agg(["mean", "std"]).reset_index()

        ax.plot(grouped["nimgs"], grouped["mean"], label=model_setting)
        # ax.fill_between(
        #     grouped["nimgs"],
        #     grouped["mean"] - grouped["std"],
        #     grouped["mean"] + grouped["std"],
        #     alpha=0.2,
        # )

    ax.set_title(metric)
    ax.set_xlabel("nimgs")
    if idx == 0:
        ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    if metric.lower() == "mse":
        ax.set_ylim(top=200)
    elif metric.lower() == "lpips":
        ax.set_ylim(top=0.1)

# Legend (shared)
handles, labels = axes3[0].get_legend_handles_labels()
if handles:
    fig3.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))

plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave top space for legend
if SAVE_FIGS:
    fig3.savefig(f"{OUTDIR}/training_efficiency_vs_nimgs.png", dpi=200, bbox_inches="tight")
plt.show()


# ----------------------------------------------------
# 5. Print summary table (unchanged)
# ----------------------------------------------------
summary = {}
for model_setting, entries in results.items():
    summary[model_setting] = {
        m: [e[m] for e in entries if e[m] is not None]
        for m in METRICS
    }

print("\n\n=== SUMMARY ===")
for model, vals in summary.items():
    print(f"\n{model}:")
    for m, list_vals in vals.items():
        if list_vals:
            print(f"  {m}: mean={sum(list_vals)/len(list_vals):.4f}, std={np.array(list_vals).std():.4f}, n={len(list_vals)}, max={max(list_vals):.4f}")
        else:
            print(f"  {m}: no data")

for model, vals in summary.items():
    print(f"& {model}", end=" ")
    for m, list_vals in vals.items():
        print(f"& ${sum(list_vals)/len(list_vals):.4f} \pm {np.array(list_vals).std():.4f}$", end=" ")
    print(r"\\")
