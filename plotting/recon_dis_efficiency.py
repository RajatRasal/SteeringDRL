import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import matplotlib as mpl


def smooth_curve(y, window=9, poly=3):
    """Savitzky–Golay smoothing with safe window handling."""
    n = len(y)
    if n < 3:
        return y  # too short to smooth
    # largest odd window <= n
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1))


def plot_recon_vs_score(
    base_dir: str,
    score_col: str = "dci",
    out_dir: str = "analysis_plots",
    cmap_name: str = "viridis",
):
    """
    Plot score_col vs MSE and vs LPIPS trajectories for all runs in base_dir.
    Curves are coloured by final score_col value.
    """
    run_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    run_dirs = sorted(run_dirs)

    if not run_dirs:
        raise RuntimeError(f"No run directories found in {base_dir}")

    # ---------- 1) compute final score per run ----------
    final_score = {}
    for run_name in run_dirs:
        metrics_path = os.path.join(base_dir, run_name, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception as e:
            print(f"Skipping {run_name}: failed to read metrics.csv ({e})")
            continue

        # NEW: skip header-only / empty metrics files
        if df is None or df.empty:
            print(f"Skipping {run_name}: metrics.csv has no rows (header-only)")
            continue

        # NEW: require nimgs for sorting / consistency
        if "nimgs" not in df.columns:
            print(f"Skipping {run_name}: metrics.csv missing 'nimgs'")
            continue

        df = df.sort_values("nimgs")

        # NEW: don't hard-fail if this run doesn't have the requested score
        if score_col not in df.columns:
            print(f"Skipping {run_name}: {score_col} not in metrics.csv columns")
            continue

        score_s = smooth_curve(df[score_col].dropna().values)
        if len(score_s) == 0:
            print(f"Skipping {run_name}: {score_col} has no entries")
            continue

        final_score[run_name] = float(score_s[-1])

    if not final_score:
        raise RuntimeError("No metrics.csv files with the requested score were found.")

    score_vals = np.array(list(final_score.values()))
    s_min, s_max = score_vals.min(), score_vals.max()
    s_range = max(s_max - s_min, 1e-8)

    def colour_for_run(name, cmap):
        v = (final_score[name] - s_min) / s_range  # 0..1
        return cmap(v)

    cmap = plt.cm.get_cmap(cmap_name)

    # ---------- 2) plot trajectories with colours ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_mse, ax_lpips = axes

    for run_name in run_dirs:
        # NEW: only plot runs that were successfully scored above
        if run_name not in final_score:
            continue

        metrics_path = os.path.join(base_dir, run_name, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue

        # NEW: skip header-only / empty metrics files
        if df is None or df.empty:
            continue

        # NEW: require nimgs for sorting / consistency
        if "nimgs" not in df.columns:
            continue

        df = df.sort_values("nimgs")

        # NEW: make sure required columns exist
        if not {"mse", "lpips", score_col}.issubset(df.columns):
            continue

        mse   = smooth_curve(df["mse"].dropna().values)
        lpips = smooth_curve(df["lpips"].dropna().values)
        score = smooth_curve(df[score_col].dropna().values)

        # NEW: guard against empty series after dropna
        if len(mse) == 0 or len(lpips) == 0 or len(score) == 0:
            continue

        # Align lengths if they differ (rare but can happen with NaN dropping)
        n = min(len(mse), len(lpips), len(score))
        mse, lpips, score = mse[:n], lpips[:n], score[:n]
        if n == 0:
            continue

        colour = colour_for_run(run_name, cmap)

        # score vs MSE trajectory (log x-scale)
        ax_mse.plot(mse, score, alpha=0.9, color=colour)
        ax_mse.scatter(mse[0],  score[0],  s=10, color=colour)   # start
        ax_mse.scatter(mse[-1], score[-1], s=25, color=colour)   # end

        # score vs LPIPS trajectory (linear x-scale)
        ax_lpips.plot(lpips, score, alpha=0.9, color=colour)
        ax_lpips.scatter(lpips[0],  score[0],  s=10, color=colour)
        ax_lpips.scatter(lpips[-1], score[-1], s=25, color=colour)

    # --- styling ---
    score_label = score_col.upper() if score_col == "dci" else score_col

    ax_mse.set_xscale("log")
    ax_mse.set_xlabel("MSE (log scale)")
    ax_mse.set_ylabel(score_label)
    ax_mse.set_title(f"{score_label} vs MSE over training")

    ax_lpips.set_xlabel("LPIPS")
    ax_lpips.set_ylabel(score_label)
    ax_lpips.set_title(f"{score_label} vs LPIPS over training")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    # --- colourbar outside ---
    norm = mpl.colors.Normalize(vmin=s_min, vmax=s_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"Final {score_label}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"recon_vs_{score_col}_coloured.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


# ---- examples ----

# base_dir = "ckpts/vdmpp/shapes3d_early_stage"
# base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech"
# base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech_dynamic"
# base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech_dynamic_rezero"
base_dir = "ckpts/vdmpp_deep/shapes3d/"

plot_recon_vs_score(base_dir, score_col="dci")
plot_recon_vs_score(base_dir, score_col="factorvae")
plot_recon_vs_score(base_dir, score_col="mig")
