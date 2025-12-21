import os
import re
import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter


def smooth_curve(y, window=9, poly=3):
    """Savitzky–Golay smoothing with safe window handling."""
    n = len(y)
    if n < 3:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1))


def parse_run_name_seedprefix(run_name: str):
    """
    Parse '{seed}v_cosine_sech_dynamic1' -> (model_class, seed).

    We treat everything AFTER the first 'v' as the model class identifier.

    Example:
        '0v_cosine_sech_dynamic1' -> model_class='v_cosine_sech_dynamic1', seed=0
        '3v_cosine_sech_dynamic1' -> model_class='v_cosine_sech_dynamic1', seed=3

    So {0-5}v_cosine_sech_dynamic1 are grouped together.
    """
    m = re.match(r"(?P<seed>\d+)v(?P<rest>.+)", run_name)
    if not m:
        # fallback: no seed prefix, treat whole name as model_class
        return run_name, None
    seed = int(m.group("seed"))
    rest = m.group("rest")
    model_class = "v" + rest
    return model_class, seed


def load_dynamic_range(config_dir: str):
    """
    Look for a single *.json config in config_dir and read dynamic_min/max.
    Returns (dynamic_min, dynamic_max) or (None, None) if not found.
    """
    try:
        cfg_paths = glob.glob(os.path.join(config_dir, "*.json"))
        if not cfg_paths:
            return None, None
        import json
        with open(cfg_paths[0], "r") as f:
            cfg = json.load(f)
        w_cfg = cfg.get("weighting", {})
        return w_cfg.get("dynamic_min"), w_cfg.get("dynamic_max")
    except Exception:
        return None, None


def plot_dis_vs_recon(
    base_dir: str,
    dis_col: str = "dci",      # disentanglement metric column
    mse_col: str = "mse",
    lpips_col: str = "lpips",
    cmap_name: str = "viridis",
    smooth: bool = True,
    out_path: str = "dis_vs_recon_avg.png",
):
    """
    For each model class (grouping {seed}v_* with same suffix),
    average trajectories over seeds and plot:

        left subplot:  dis vs mse
        right subplot: dis vs lpips

    One line per model class:
      - colour: final dis_col (mean across seeds)
      - linestyle: unique per model class
    """
    run_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {base_dir}")

    # ---------- group runs by model_class ----------
    groups: Dict[str, List[str]] = {}
    for run_name in run_dirs:
        model_class, seed = parse_run_name_seedprefix(run_name)
        groups.setdefault(model_class, []).append(run_name)

    # ---------- load metrics per run ----------
    per_run_df: Dict[str, pd.DataFrame] = {}
    for run_name in run_dirs:
        metrics_path = os.path.join(base_dir, run_name, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        try:
            df = pd.read_csv(metrics_path)
    
            # NEW: skip header-only / empty metrics files
            if df is None or df.empty:
                print(f"Skipping {run_name}: metrics.csv has no rows (header-only)")
                continue
            
            # NEW: skip files missing nimgs (can't sort / align)
            if "nimgs" not in df.columns:
                print(f"Skipping {run_name}: metrics.csv missing 'nimgs' column")
                continue
            
            df = df.sort_values("nimgs")
            per_run_df[run_name] = df
    
        except Exception as e:
            print(f"Skipping {run_name}: failed to read metrics.csv ({e})")
            continue
        
    # ---------- final DIS per model_class (mean over seeds) ----------
    final_dis_by_model: Dict[str, float] = {}
    for model_class, runs in groups.items():
        scores = []
        for r in runs:
            df = per_run_df.get(r)
            if df is None or dis_col not in df.columns:
                continue
            y = df[dis_col].values
            if smooth:
                y = smooth_curve(y)
            scores.append(float(y[-1]))
        if scores:
            final_dis_by_model[model_class] = float(np.mean(scores))

    if not final_dis_by_model:
        raise RuntimeError("No disentanglement scores found; check column names.")

    dis_vals = np.array(list(final_dis_by_model.values()))
    d_min, d_max = dis_vals.min(), dis_vals.max()
    d_range = max(d_max - d_min, 1e-8)

    def colour_for_model(model_class, cmap):
        v = (final_dis_by_model[model_class] - d_min) / d_range
        return cmap(v)

    cmap = plt.cm.get_cmap(cmap_name)
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax_mse, ax_lpips = axes

    # ---------- build DIS vs recon trajectories per model class ----------
    for i, (model_class, runs) in enumerate(sorted(groups.items())):
        dis_trajs = []
        mse_trajs = []
        lpips_trajs = []
        nimgs_ref = None

        for r in runs:
            df = per_run_df.get(r)
            if df is None:
                continue
            if not {dis_col, mse_col, lpips_col}.issubset(df.columns):
                continue

            x = df["nimgs"].values
            dis = df[dis_col].values
            mse = df[mse_col].values
            lp = df[lpips_col].values

            if smooth:
                dis = smooth_curve(dis)
                mse = smooth_curve(mse)
                lp = smooth_curve(lp)

            if nimgs_ref is None:
                nimgs_ref = x
            else:
                # interpolate to reference nimgs grid if needed
                if not np.array_equal(x, nimgs_ref):
                    dis = np.interp(nimgs_ref, x, dis)
                    mse = np.interp(nimgs_ref, x, mse)
                    lp = np.interp(nimgs_ref, x, lp)

            dis_trajs.append(dis)
            mse_trajs.append(mse)
            lpips_trajs.append(lp)

        if not dis_trajs or nimgs_ref is None:
            continue

        dis_trajs = np.stack(dis_trajs, axis=0)
        mse_trajs = np.stack(mse_trajs, axis=0)
        lpips_trajs = np.stack(lpips_trajs, axis=0)

        mean_dis = dis_trajs.mean(axis=0)
        mean_mse = mse_trajs.mean(axis=0)
        mean_lp = lpips_trajs.mean(axis=0)

        colour = colour_for_model(model_class, cmap)
        linestyle = linestyles[i % len(linestyles)]

        cfg_dir = os.path.join(base_dir, runs[0], "configs")
        dyn_min, dyn_max = load_dynamic_range(cfg_dir)
        if dyn_min is not None and dyn_max is not None:
            label = f"{model_class}  [{dyn_min}, {dyn_max}]"
        else:
            label = model_class

        # left: DIS vs MSE
        ax_mse.plot(
            mean_mse,
            mean_dis,
            linestyle=linestyle,
            color=colour,
            label=label,
            linewidth=2,
        )
        # right: DIS vs LPIPS
        ax_lpips.plot(
            mean_lp,
            mean_dis,
            linestyle=linestyle,
            color=colour,
            label=label,
            linewidth=2,
        )

    ax_mse.set_xlabel(mse_col)
    ax_lpips.set_xlabel(lpips_col)
    ax_mse.set_ylabel(dis_col)

    ax_mse.set_title(f"{dis_col} vs {mse_col}")
    ax_lpips.set_title(f"{dis_col} vs {lpips_col}")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)

    # shared legend at bottom
    handles, labels = ax_mse.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=2)
    fig.subplots_adjust(bottom=0.25, right=0.88)

    # colourbar for final DIS
    norm = mpl.colors.Normalize(vmin=d_min, vmax=d_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.25, 0.02, 0.6])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f"final {dis_col}")

    fig.suptitle("Disentanglement vs reconstruction (mean over seeds)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    # base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech_dynamic"
    # base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech_dynamic_rezero"
    # base_dir = "ckpts/vdmpp/shapes3d_early_stage_sech_dynamic_rezero_sharpening"
    base_dir = "ckpts/vdmpp_deep/shapes3d/"
    # colour by final DCI, plot DCI trajectory
    plot_dis_vs_recon(
        base_dir,
        dis_col="dci",      # or "factorvae", "mig", etc.
        mse_col="mse",
        lpips_col="lpips",
        out_path="analysis_plots/avg_dci_vs_nimgs.pdf",
    )
    plot_dis_vs_recon(
        base_dir,
        dis_col="factorvae",      # or "factorvae", "mig", etc.
        mse_col="mse",
        lpips_col="lpips",
        out_path="analysis_plots/avg_factorvae_vs_nimgs.pdf",
    )
    plot_dis_vs_recon(
        base_dir,
        dis_col="mig",      # or "factorvae", "mig", etc.
        mse_col="mse",
        lpips_col="lpips",
        out_path="analysis_plots/avg_mig_vs_nimgs.pdf",
    )
