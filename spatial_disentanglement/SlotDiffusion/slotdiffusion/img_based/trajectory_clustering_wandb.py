import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.signal import savgol_filter


RUN_METRIC_RE = re.compile(r"^(?P<run>.+?) - (?P<metric>.+?)(?:__(?P<suffix>MIN|MAX))?$")
RUN_BASE_RE = re.compile(r"^(?P<model>.+)-dup(?P<dup>\d+)-(?P<job>\d+)$")

DEFAULT_METRICS = ["val/fari", "val/miou", "val/mbo"]
METRIC_SPECS = {
    "val/fari": dict(label="FG-ARI", lb=0.5, ub=0.9, ymin=0.0, ymax=1.0),
    "val/miou": dict(label="mIoU", lb=0.5, ub=0.7, ymin=0.0, ymax=0.7),
    "val/mbo": dict(label="mBO", lb=0.5, ub=0.75, ymin=0.0, ymax=0.75),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot SlotDiffusion trajectory curves from W&B CSV exports for two models. "
            "Each CSV may contain one or more metrics; __MIN/__MAX columns are ignored."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_paths",
        nargs="+",
        required=True,
        help="One or more W&B export CSV files.",
    )
    parser.add_argument(
        "--models",
        nargs=2,
        required=True,
        metavar=("MODEL_A", "MODEL_B"),
        help="Base model names to compare, e.g. foo and bar for foo-dup1-<jobid>.",
    )
    parser.add_argument(
        "--labels",
        nargs=2,
        metavar=("LABEL_A", "LABEL_B"),
        help="Optional legend labels for the two models.",
    )
    parser.add_argument(
        "--linestyles",
        nargs=2,
        default=["-", "--"],
        metavar=("STYLE_A", "STYLE_B"),
        help="Line styles for the two models.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to plot, in order. Defaults to val/fari val/miou val/mbo.",
    )
    parser.add_argument(
        "--x-csv",
        dest="x_csv_paths",
        nargs="+",
        default=None,
        help="Optional W&B export CSV(s) that provide the x-axis metric.",
    )
    parser.add_argument(
        "--x-metric",
        default=None,
        help="Metric name to use from --x-csv, e.g. val/dpm_sample_img_recon_loss.",
    )
    parser.add_argument(
        "--x-label",
        default=None,
        help="Override x-axis label. By default this is inferred from --x-metric.",
    )
    parser.add_argument(
        "--batch-size",
        type=float,
        default=64.0,
        help="Batch size used to convert steps to images seen.",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=3,
        help="Reference point index used for the square marker and legend.",
    )
    parser.add_argument(
        "--start-step",
        type=float,
        default=0.0,
        help="Synthetic start step prepended before the first logged point.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Defaults to analysis_plots/<modelA>_vs_<modelB>_traj.pdf",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the plot interactively.",
    )
    return parser.parse_args()


def configure_style():
    for style_name in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"]:
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "-",
            "grid.linewidth": 1.35,
            "axes.edgecolor": "#cccccc",
            "axes.linewidth": 1.0,
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
        }
    )


def smooth_curve(y: np.ndarray, window: int = 9, poly: int = 3) -> np.ndarray:
    n = len(y)
    if n < 3:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1))


def parse_float(value: str):
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "nan":
        return None
    return float(value)


def base_model_name(run_name: str) -> str:
    match = RUN_BASE_RE.match(run_name)
    return match.group("model") if match else run_name


def run_sort_key(run_name: str) -> Tuple[int, str]:
    match = RUN_BASE_RE.match(run_name)
    if not match:
        return (10**9, run_name)
    return (int(match.group("dup")), run_name)


def human_images(num_images: float) -> str:
    if num_images >= 1_000_000:
        return f"{num_images / 1_000_000:.1f}M"
    if num_images >= 1_000:
        return f"{num_images / 1_000:.0f}K"
    return f"{num_images:.0f}"


def image_formatter(value, _pos):
    return human_images(value)


def default_x_label(metric_name: Optional[str]) -> str:
    if not metric_name:
        return "Images Seen"
    if metric_name == "val/dpm_sample_img_recon_loss":
        return "Reconstruction (Pixel-Level Loss)"
    if "lpips" in metric_name.lower():
        return "Reconstruction (LPIPS)"
    if "recon" in metric_name.lower():
        return f"Reconstruction ({metric_name.split('/')[-1]})"
    return metric_name


def read_wandb_csv(path: str, start_step: float) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    series = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"steps": [], "values": []})))

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        columns = []
        for field in fieldnames:
            if field == "Step":
                continue
            match = RUN_METRIC_RE.match(field)
            if not match:
                continue
            if match.group("suffix") is not None:
                continue
            run_name = match.group("run")
            metric_name = match.group("metric")
            model_name = base_model_name(run_name)
            columns.append((field, metric_name, model_name, run_name))

        for row in reader:
            step = parse_float(row.get("Step"))
            if step is None:
                continue
            for field, metric_name, model_name, run_name in columns:
                value = parse_float(row.get(field))
                if value is None:
                    continue
                record = series[metric_name][model_name][run_name]
                record["steps"].append(step)
                record["values"].append(value)

    output = defaultdict(lambda: defaultdict(dict))
    for metric_name, metric_models in series.items():
        for model_name, runs in metric_models.items():
            for run_name, record in runs.items():
                steps = np.asarray(record["steps"], dtype=float)
                values = np.asarray(record["values"], dtype=float)
                if steps.size == 0:
                    continue
                if steps[0] > start_step:
                    steps = np.insert(steps, 0, start_step)
                    values = np.insert(values, 0, values[0])
                output[metric_name][model_name][run_name] = (steps, values)
    return output


def merge_metric_data(csv_paths: List[str], start_step: float):
    merged = defaultdict(lambda: defaultdict(dict))
    for path in csv_paths:
        csv_data = read_wandb_csv(path, start_step=start_step)
        for metric_name, metric_models in csv_data.items():
            for model_name, runs in metric_models.items():
                merged[metric_name][model_name].update(runs)
    return merged


def infer_single_metric_name(metric_data) -> Optional[str]:
    metrics = sorted(metric_data.keys())
    if len(metrics) == 1:
        return metrics[0]
    return None


def align_series(
    x_series: Tuple[np.ndarray, np.ndarray], y_series: Tuple[np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_steps, x_values = x_series
    y_steps, y_values = y_series

    common_steps, x_idx, y_idx = np.intersect1d(x_steps, y_steps, return_indices=True)
    if common_steps.size == 0:
        raise ValueError("No common steps found between x-axis and y-axis series.")
    return common_steps, x_values[x_idx], y_values[y_idx]


def build_figure(n_metrics: int):
    width_ratios = []
    for idx in range(n_metrics):
        width_ratios.extend([1.0, 0.06])
        if idx != n_metrics - 1:
            width_ratios.append(0.065)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(width_ratios),
        figsize=(20 if n_metrics == 3 else 6.7 * n_metrics, 5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    plot_axes = []
    cbar_axes = []
    axis_idx = 0
    for idx in range(n_metrics):
        plot_axes.append(axes[axis_idx])
        cbar_axes.append(axes[axis_idx + 1])
        axis_idx += 2
        if idx != n_metrics - 1:
            axes[axis_idx].set_axis_off()
            axis_idx += 1
    return fig, axes, plot_axes, cbar_axes


def main():
    args = parse_args()
    configure_style()

    labels = args.labels or args.models
    model_styles = list(zip(args.models, labels, args.linestyles))

    metric_data = merge_metric_data(args.csv_paths, start_step=args.start_step)
    available_metrics = [metric for metric in args.metrics if metric in metric_data]
    if not available_metrics:
        raise ValueError(
            f"None of the requested metrics {args.metrics} were found in {args.csv_paths}. "
            f"Available metrics: {sorted(metric_data.keys())}"
        )

    x_metric_data = None
    x_metric_name = None
    x_label = "Images Seen"
    if args.x_csv_paths:
        x_metric_data = merge_metric_data(args.x_csv_paths, start_step=args.start_step)
        x_metric_name = args.x_metric or infer_single_metric_name(x_metric_data)
        if x_metric_name is None:
            raise ValueError(
                "Could not infer a unique x-axis metric from --x-csv. "
                f"Available metrics: {sorted(x_metric_data.keys())}. Please pass --x-metric."
            )
        if x_metric_name not in x_metric_data:
            raise ValueError(
                f"x-axis metric '{x_metric_name}' was not found in {args.x_csv_paths}. "
                f"Available metrics: {sorted(x_metric_data.keys())}"
            )
        x_label = args.x_label or default_x_label(x_metric_name)
    elif args.x_label:
        x_label = args.x_label

    for model_name in args.models:
        present = any(model_name in metric_data[metric] for metric in available_metrics)
        if not present:
            raise ValueError(f"Model '{model_name}' was not found in the provided CSV files.")
        if x_metric_data is not None and model_name not in x_metric_data[x_metric_name]:
            raise ValueError(
                f"Model '{model_name}' was not found in x-axis metric '{x_metric_name}'."
            )

    fig, axes, plot_axes, cbar_axes = build_figure(len(available_metrics))
    cmap = plt.get_cmap("coolwarm_r")
    marker_color = "mediumseagreen"

    reference_images = None
    end_images = None

    for idx, metric_name in enumerate(available_metrics):
        spec = METRIC_SPECS.get(
            metric_name,
            dict(label=metric_name, lb=0.0, ub=1.0, ymin=0.0, ymax=1.0),
        )
        ax = plot_axes[idx]
        cax = cbar_axes[idx]
        norm = plt.Normalize(vmin=spec["lb"], vmax=spec["ub"])

        metric_has_any = False
        for model_name, model_label, linestyle in model_styles:
            runs = metric_data[metric_name].get(model_name, {})
            for run_name in sorted(runs, key=run_sort_key):
                steps, values = runs[run_name]
                if x_metric_data is not None:
                    x_runs = x_metric_data[x_metric_name].get(model_name, {})
                    if run_name not in x_runs:
                        continue
                    aligned_steps, x_values, y_values = align_series(x_runs[run_name], runs[run_name])
                    x = smooth_curve(x_values)
                    y = smooth_curve(y_values)
                    stage_images = aligned_steps * args.batch_size
                else:
                    x = steps * args.batch_size
                    y = smooth_curve(values)
                    stage_images = steps * args.batch_size
                metric_has_any = True

                ax.plot(
                    x,
                    y,
                    c=cmap(norm(values[-1])),
                    linestyle=linestyle,
                    linewidth=2,
                    alpha=0.95,
                )

                ref_idx = min(args.ref_index, len(x) - 1)
                ax.scatter(
                    x[0],
                    y[0],
                    c=marker_color,
                    marker="o",
                    s=50,
                    edgecolor="black",
                    linewidth=2,
                    zorder=50,
                )
                ax.scatter(
                    x[ref_idx],
                    y[ref_idx],
                    marker="P",
                    s=80,
                    c=marker_color,
                    edgecolor="black",
                    linewidths=1,
                    zorder=50,
                )
                ax.scatter(
                    x[-1],
                    y[-1],
                    c=marker_color,
                    marker="^",
                    s=50,
                    edgecolor="black",
                    linewidth=2,
                    zorder=50,
                )

                if reference_images is None:
                    reference_images = stage_images[ref_idx]
                if end_images is None or stage_images[-1] > end_images:
                    end_images = stage_images[-1]

        if not metric_has_any:
            raise ValueError(f"No runs found for metric '{metric_name}' and models {args.models}.")

        ax.set_ylim(spec["ymin"], spec["ymax"])
        ax.set_xlabel(x_label, fontsize=15)
        ax.set_ylabel(f"Segmentation ({spec['label']})", fontsize=15)
        ax.tick_params(axis="y", left=True, labelleft=True, length=5, width=1, labelsize=15)
        ax.tick_params(axis="x", bottom=True, labelbottom=True, length=5, width=1, labelsize=15)
        if x_metric_data is None:
            ax.xaxis.set_major_formatter(FuncFormatter(image_formatter))

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(f"Final {spec['label']}", fontsize=15)
        cbar.ax.tick_params(labelsize=15)
        cbar.outline.set_linewidth(1.5)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=marker_color,
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=10,
            label="Start (0 imgs)",
        ),
        Line2D(
            [0],
            [0],
            marker="P",
            linestyle="None",
            markerfacecolor=marker_color,
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=10,
            label=f"Ref. ({human_images(reference_images or 0)} imgs)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="None",
            markerfacecolor=marker_color,
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=10,
            label=f"End ({human_images(end_images or 0)} imgs)",
        ),
    ]
    for _model_name, model_label, linestyle in model_styles:
        legend_handles.append(
            Line2D([0], [0], color="black", linestyle=linestyle, linewidth=2, label=model_label)
        )

    plot_axes[-1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        fontsize=12,
    )

    if args.output is None:
        basename = f"{args.models[0]}_vs_{args.models[1]}_traj.pdf"
        output_path = os.path.join("analysis_plots", basename)
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
