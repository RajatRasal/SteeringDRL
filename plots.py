import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # NEW
from collections import defaultdict

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
ROOT = "ckpts/vdmpp/shapes3d_low_discrep"
METRICS = ["dci", "factorvae", "mig", "mse", "lpips"]
SAVE_FIGS = True
OUTDIR = "analysis_plots"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------------------------------
# 1. Walk the directory and gather metrics per model_setting
# ----------------------------------------------------
results = defaultdict(list)        # final values
curves = defaultdict(list)         # NEW: full curves per run

for folder in os.listdir(ROOT):
    folder_path = os.path.join(ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    # expect folder names like "0shapes3d_v_cosine_sech"
    try:
        seed = int(folder[0])
        model_setting = folder[1:]
    except:
        # skip folders not matching the pattern
        continue

    metrics_path = os.path.join(folder_path, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"Skipping {folder}, no metrics.csv")
        continue

    df = pd.read_csv(metrics_path)

    # store final values
    entry = {}
    for m in METRICS:
        if m in df.columns:
            entry[m] = df[m].iloc[-1]  # final value
        else:
            entry[m] = None
    results[model_setting].append(entry)

    # NEW: store full curve (nimgs + metrics) for training-efficiency plot
    needed_cols = ["nimgs"] + [m for m in METRICS if m in df.columns]
    curves[model_setting].append(df[needed_cols].copy())


# ----------------------------------------------------
# 2. Single figure: boxplots (1 row, N metrics)
# ----------------------------------------------------
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

# create subplots: one row, one column per metric
n_metrics = len(valid_metrics)
fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 8), squeeze=False)

for idx, (metric, data) in enumerate(zip(valid_metrics, metric_data_list)):
    ax = axes[0, idx]

    labels = list(data.keys())
    values = [data[label] for label in labels]

    ax.boxplot(values, labels=labels)
    ax.set_title(metric)
    ax.tick_params(axis='x', rotation=90)
    if idx == 0:
        ax.set_ylabel("Score")

plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(f"{OUTDIR}/all_metrics_boxplots_row.png", dpi=200)
plt.show()


# ----------------------------------------------------
# 3. Mean + Std Error Bar PLOTS (1 row × N metrics) — no connecting lines
# ----------------------------------------------------
fig2, axes2 = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6), squeeze=False)

for idx, (metric, data) in enumerate(zip(valid_metrics, metric_data_list)):
    ax = axes2[0, idx]

    labels = list(data.keys())
    means = [pd.Series(v).mean() for v in data.values()]
    stds  = [pd.Series(v).std()  for v in data.values()]

    ax.errorbar(
        x=range(len(labels)),
        y=means,
        yerr=stds,
        fmt='o',              # dots only
        linestyle='none',     # absolutely no connecting lines
        capsize=5,
        markersize=6
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)

    ax.set_title(metric + " (mean ± std)")
    if idx == 0:
        ax.set_ylabel("Score")

plt.tight_layout()
if SAVE_FIGS:
    fig2.savefig(f"{OUTDIR}/all_metrics_mean_std_row.png", dpi=200)
plt.show()


# ----------------------------------------------------
# 4. Training efficiency vs nimgs (1 row × N metrics)
#    Uses groupby over nimgs to handle different run lengths
# ----------------------------------------------------
fig3, axes3 = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6), squeeze=False)

for idx, metric in enumerate(valid_metrics):
    ax = axes3[0, idx]

    for model_setting, runs in curves.items():
        # Only keep runs where this metric is present
        metric_runs = [r[["nimgs", metric]].dropna(subset=[metric]) 
                       for r in runs if metric in r.columns]
        if not metric_runs:
            continue

        # Concatenate all seeds into one long DF:
        # columns: nimgs, metric
        df_all = pd.concat(metric_runs, ignore_index=True)

        # Group by nimgs and compute mean/std across seeds
        grouped = df_all.groupby("nimgs")[metric].agg(["mean", "std"]).reset_index()

        nimgs = grouped["nimgs"].values
        mean_curve = grouped["mean"].values
        std_curve = grouped["std"].values

        ax.plot(nimgs, mean_curve, label=model_setting)
        # If you want std shading, uncomment:
        # ax.fill_between(nimgs,
        #                 mean_curve - std_curve,
        #                 mean_curve + std_curve,
        #                 alpha=0.2)

        if metric == "mse":
            ax.set_ylim(15, 40)
        if metric == "lpips":
            ax.set_ylim(0.015, 0.05)

    ax.set_title(f"{metric} vs nimgs")
    ax.set_xlabel("images seen (nimgs)")
    if idx == 0:
        ax.set_ylabel(metric)

    ax.grid(True, alpha=0.3)

# Legend (shared)
handles, labels = axes3[0, 0].get_legend_handles_labels()
if handles:
    fig3.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))

plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave top space for legend
if SAVE_FIGS:
    fig3.savefig(f"{OUTDIR}/training_efficiency_vs_nimgs.png", dpi=200)
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
            print(f"  {m}: mean={sum(list_vals)/len(list_vals):.4f}, n={len(list_vals)}")
        else:
            print(f"  {m}: no data")
