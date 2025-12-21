import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================
# Config: which model names count as monotonic
# ============================
MONOTONIC_KEYS = ["sigmoid", "exp", "edm_mono"]  # user-specified


# ============================
# Utilities
# ============================
def parse_run_name_seedprefix(run_name: str) -> Tuple[str, Optional[int]]:
    """
    Parse '{seed}v_cosine_sech_dynamic1' -> (model_class, seed).
    Everything AFTER the first 'v' is treated as model_class identifier.
    """
    m = re.match(r"(?P<seed>\d+)v(?P<rest>.+)", str(run_name))
    if not m:
        return str(run_name), None
    seed = int(m.group("seed"))
    model_class = "v" + m.group("rest")
    return model_class, seed


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        return None if df is None or df.empty else df
    except Exception:
        return None


def choose_time_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for c in preferred:
        if c in df.columns:
            return c
    return None


def rolling_smooth_inplace(df: pd.DataFrame, col: str, k: int) -> None:
    if k is None or k <= 1 or col not in df.columns:
        return
    df[col] = df[col].rolling(window=k, min_periods=1).mean()


def is_monotonic_model(model_class: str) -> bool:
    s = str(model_class).lower()
    return any(k in s for k in MONOTONIC_KEYS)


def linestyle_for_model(model_class: str) -> str:
    # monotonic => dotted, non-monotonic => solid
    return "-" if is_monotonic_model(model_class) else "-"


# ============================
# Core plotting helper
# ============================
def plot_all_models_single_axis(
    merged: pd.DataFrame,
    y_col: str,
    color_metric: str,
    time_col_name: str,
    out_path: str,
    invert_cmap: bool = False,
    smooth_k: int = 1,
    show_band: bool = True,
):
    """
    One axes:
      - one line per model_class (mean over seeds)
      - y = y_col
      - colour = mean final color_metric per model_class
      - linestyle = dotted for monotonic (sigmoid/exp/edm_mono), solid otherwise
    """
    if color_metric not in merged.columns:
        print(f"Skipping {out_path}: '{color_metric}' not found.")
        return

    # optional smoothing per run (only for y_col; keeps things crisp but stable)
    if smooth_k and smooth_k > 1:
        merged = (
            merged.sort_values("_time")
            .groupby(["model_class", "seed", "run_name"], dropna=False, group_keys=False)
            .apply(lambda g: g.assign(**{y_col: g[y_col].rolling(smooth_k, min_periods=1).mean()}))
            .reset_index(drop=True)
        )

    final_by_model = merged.groupby("model_class")[color_metric].mean().dropna()
    if final_by_model.empty:
        print(f"Skipping {out_path}: no valid '{color_metric}' values.")
        return

    vmin, vmax = float(final_by_model.min()), float(final_by_model.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    cmap = plt.cm.viridis_r if invert_cmap else plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 5))

    for model_class, g in merged.groupby("model_class"):
        if model_class not in final_by_model:
            continue

        agg = (
            g.groupby("_time")
            .agg(mean=(y_col, "mean"), std=(y_col, "std"))
            .reset_index()
            .sort_values("_time")
        )

        colour = cmap(norm(final_by_model[model_class]))
        ls = linestyle_for_model(model_class)

        ax.plot(
            agg["_time"],
            agg["mean"],
            color=colour,
            linestyle=ls,
            linewidth=2.5,
            label=model_class,
        )

        if show_band:
            ax.fill_between(
                agg["_time"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                color=colour,
                alpha=0.12,
            )

    ax.set_title(f"{y_col} (model-class mean) — colour: final {color_metric} | linestyle: monotonic dotted")
    ax.set_xlabel(time_col_name)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)

    # colorbar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"mean final {color_metric}")

    # legend
    ax.legend(fontsize=7, ncol=2, frameon=False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================
# Main: global-only plots (no per-model-class individual figures)
# ============================
def plot_gradients_global(
    base_dir: str,
    out_dir: str = "analysis_plots/grad_debug",
    time_preference: List[str] = ("nimgs", "step", "global_step", "iter"),
    enc_col: str = "window_enc_mean",
    dec_col: str = "window_dec_mean",
    smooth_k: int = 5,
    dis_color_metric: str = "dci",
    recon_color_metric: str = "mse",
):
    """
    Assumes:
      base_dir/<run_name>/{debug.csv, metrics.csv}

    Produces ONLY global plots (one line per model class; mean over seeds):
      - Encoder grad coloured by DCI
      - Decoder grad coloured by DCI
      - Decoder grad coloured by MSE
      - Enc/Dec ratio coloured by DCI
      - Enc/Dec ratio coloured by MSE

    Linestyle convention:
      - dotted ":" for monotonic (sigmoid, exp, edm_mono)
      - solid "-" for non-monotonic
    """
    os.makedirs(out_dir, exist_ok=True)

    debug_rows = []
    final_rows = []
    chosen_time_col: Optional[str] = None

    print(os.listdir(base_dir))

    run_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    run_dirs = sorted(run_dirs)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {base_dir}")

    for run_name in run_dirs:
        run_path = os.path.join(base_dir, run_name)
        df_debug = safe_read_csv(os.path.join(run_path, "debug.csv"))
        df_metrics = safe_read_csv(os.path.join(run_path, "metrics.csv"))
        if df_debug is None or df_metrics is None:
            continue
        if enc_col not in df_debug.columns or dec_col not in df_debug.columns:
            continue

        model_class, seed = parse_run_name_seedprefix(run_name)

        # choose time column (prefer debug)
        if chosen_time_col is None:
            tc = choose_time_col(df_debug, list(time_preference))
            if tc is None:
                tc = choose_time_col(df_metrics, list(time_preference))
            chosen_time_col = tc

        if chosen_time_col is None or chosen_time_col not in df_debug.columns:
            continue

        df_debug = df_debug.sort_values(chosen_time_col).copy()
        df_debug["_time"] = df_debug[chosen_time_col].astype(float)
        df_debug["model_class"] = model_class
        df_debug["seed"] = seed
        df_debug["run_name"] = run_name

        # smooth encoder/decoder in debug (per run)
        if smooth_k and smooth_k > 1:
            df_debug = df_debug.sort_values("_time")
            rolling_smooth_inplace(df_debug, enc_col, smooth_k)
            rolling_smooth_inplace(df_debug, dec_col, smooth_k)

        df_debug["grad_ratio"] = df_debug[enc_col] / (df_debug[dec_col] + 1e-8)
        debug_rows.append(df_debug[["model_class", "seed", "run_name", "_time", enc_col, dec_col, "grad_ratio"]])

        # final metrics (from last row of metrics.csv)
        mtc = choose_time_col(df_metrics, list(time_preference))
        if mtc is not None:
            df_metrics = df_metrics.sort_values(mtc)
        last = df_metrics.tail(1)

        row = {"model_class": model_class, "seed": seed, "run_name": run_name}
        # only grab what we need for colouring
        if dis_color_metric in last.columns:
            row[dis_color_metric] = float(last[dis_color_metric].values[0])
        else:
            row[dis_color_metric] = np.nan

        if recon_color_metric in last.columns:
            row[recon_color_metric] = float(last[recon_color_metric].values[0])
        else:
            row[recon_color_metric] = np.nan

        final_rows.append(row)

    if not debug_rows:
        raise RuntimeError(f"No usable debug.csv found under {base_dir}")

    df_debug_all = pd.concat(debug_rows, ignore_index=True)
    df_final = pd.DataFrame(final_rows).drop_duplicates(subset=["model_class", "seed", "run_name"])

    merged = df_debug_all.merge(df_final, on=["model_class", "seed", "run_name"], how="inner")

    time_name = chosen_time_col if chosen_time_col is not None else "time"

    # ------------------------
    # Encoder gradients coloured by DCI (requested missing bit)
    # ------------------------
    plot_all_models_single_axis(
        merged=merged,
        y_col=enc_col,
        color_metric=dis_color_metric,
        time_col_name=time_name,
        out_path=os.path.join(out_dir, f"ALL__encoder_grad_colored_by_{dis_color_metric}.png"),
        invert_cmap=False,
        smooth_k=1,          # already smoothed in df_debug per run; keep 1 here
        show_band=True,
    )

    # ------------------------
    # Decoder gradients coloured by DCI
    # ------------------------
    plot_all_models_single_axis(
        merged=merged,
        y_col=dec_col,
        color_metric=dis_color_metric,
        time_col_name=time_name,
        out_path=os.path.join(out_dir, f"ALL__decoder_grad_colored_by_{dis_color_metric}.png"),
        invert_cmap=False,
        smooth_k=1,
        show_band=True,
    )

    # ------------------------
    # Decoder gradients coloured by MSE (invert so "better" (lower) maps consistently)
    # ------------------------
    plot_all_models_single_axis(
        merged=merged,
        y_col=dec_col,
        color_metric=recon_color_metric,
        time_col_name=time_name,
        out_path=os.path.join(out_dir, f"ALL__decoder_grad_colored_by_{recon_color_metric}.png"),
        invert_cmap=True,
        smooth_k=1,
        show_band=True,
    )

    # ------------------------
    # Enc/Dec ratio coloured by DCI
    # ------------------------
    plot_all_models_single_axis(
        merged=merged,
        y_col="grad_ratio",
        color_metric=dis_color_metric,
        time_col_name=time_name,
        out_path=os.path.join(out_dir, f"ALL__enc_dec_ratio_colored_by_{dis_color_metric}.png"),
        invert_cmap=False,
        smooth_k=1,
        show_band=True,
    )

    # ------------------------
    # Enc/Dec ratio coloured by MSE
    # ------------------------
    plot_all_models_single_axis(
        merged=merged,
        y_col="grad_ratio",
        color_metric=recon_color_metric,
        time_col_name=time_name,
        out_path=os.path.join(out_dir, f"ALL__enc_dec_ratio_colored_by_{recon_color_metric}.png"),
        invert_cmap=True,
        smooth_k=1,
        show_band=True,
    )


if __name__ == "__main__":
    # base_dir="ckpts/vdmpp_deep/shapes3d/",
    base_dir="ckpts/vdmpp_deep_longer_stable/shapes3d/"
    plot_gradients_global(
        base_dir=base_dir,
        out_dir="analysis_plots/grad_debug",
        time_preference=["nimgs", "step", "global_step", "iter"],
        enc_col="window_enc_mean",
        dec_col="window_dec_mean",
        smooth_k=5,
        dis_color_metric="mig",   # colour by DCI for disentanglement
        recon_color_metric="mse", # colour by MSE for reconstruction
    )
