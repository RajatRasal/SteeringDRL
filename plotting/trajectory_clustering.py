import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA


# ----------------- config -----------------

base_dir = "ckpts/vdmpp/shapes3d_early_stage"
out_dir = "analysis_plots"

metrics = ["dci", "factorvae", "mig", "mse", "lpips"]   # 5D trajectory
T = 200
n_clusters = 2
smooth_window = 9
smooth_poly = 3

os.makedirs(out_dir, exist_ok=True)


# ----------------- helpers -----------------

def smooth_curve(y, window=9, poly=3):
    n = len(y)
    if n < 3:
        return y
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1))


MAX_ROWS   = 25   # use only first N rows of metrics.csv
MIN_POINTS = 10    # require at least this many points after cleaning


def load_multi_trajectory(run_path, T=200):
    """
    Load metrics.csv and return a (T, D) trajectory where D=len(metrics).
    Uses only the first MAX_ROWS rows, drops NaNs, and skips too-short runs.
    """
    mpath = os.path.join(run_path, "metrics.csv")
    if not os.path.exists(mpath):
        return None

    df = pd.read_csv(mpath).sort_values("nimgs")

    # keep only first N rows
    df = df.iloc[:MAX_ROWS]

    # ensure all required metrics exist
    for col in metrics:
        if col not in df.columns:
            raise KeyError(f"{col} missing in metrics.csv columns: {df.columns}")

    # drop rows where any of the metrics is NaN
    df = df.dropna(subset=metrics)

    # also need nimgs for time; drop rows where nimgs is NaN just in case
    df = df.dropna(subset=["nimgs"])

    if len(df) < MIN_POINTS:
        # too few valid points to build a stable trajectory
        return None

    x = df["nimgs"].values.astype(float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

    t_target = np.linspace(0., 1., T)

    traj_dim = []
    for col in metrics:
        y = df[col].values.astype(float)
        y = smooth_curve(y)

        if col == "mse":
            y = np.log(y + 1e-8)  # stabilise scale

        y_interp = np.interp(t_target, x_norm, y)
        traj_dim.append(y_interp)

    # stack to (T, D)
    return np.stack(traj_dim, axis=-1)


# ----------------- load all trajectories -----------------

run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
run_dirs = sorted(run_dirs)

trajectories = []
run_names = []

for rn in run_dirs:
    traj = load_multi_trajectory(os.path.join(base_dir, rn), T)
    if traj is None:
        continue
    trajectories.append(traj)
    run_names.append(rn)

trajectories = np.stack(trajectories, axis=0)   # shape (N, T, D)
N, _, D = trajectories.shape
print(f"Loaded {N} trajectories of shape (T={T}, D={D})")


# ----------------- DTW KMeans -----------------

dtw_km = TimeSeriesKMeans(
    n_clusters=n_clusters,
    metric="dtw",
    max_iter=50,
    random_state=0,
    n_init=4,
)
clusters = dtw_km.fit_predict(trajectories)
centroids = dtw_km.cluster_centers_  # shape (k, T, D)

print("Cluster counts:", np.bincount(clusters))


# ----------------- Visualise each cluster per dimension -----------------

t_axis = np.linspace(0., 1., T)

for k in range(n_clusters):
    for dim, col in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i in range(N):
            if clusters[i] == k:
                ax.plot(t_axis, trajectories[i][:, dim], alpha=0.25)

        ax.plot(t_axis, centroids[k][:, dim], color="black", linewidth=3, label="centroid")

        ax.set_title(f"Cluster {k} – {col} trajectory")
        ax.set_xlabel("normalised training time")
        ax.set_ylabel(col)
        ax.legend()
        fig.tight_layout()

        fname = f"traj_cluster{k}_{col}.pdf"
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)


# ----------------- PCA embedding for visualising cluster separation -----------------

# Flatten each trajectory (T*D) for PCA
X_flat = trajectories.reshape(N, T*D)

pca = PCA(n_components=2)
emb = pca.fit_transform(X_flat)

fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(emb[:, 0], emb[:, 1], c=clusters, cmap="tab10", s=50)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA of 5D trajectories (colour = DTW cluster)")
fig.colorbar(scatter, ax=ax, label="cluster")
fig.tight_layout()

fig.savefig(os.path.join(out_dir, "pca_multidim_clusters.pdf"), bbox_inches="tight")
plt.close(fig)

print("Saved PCA embedding.")
