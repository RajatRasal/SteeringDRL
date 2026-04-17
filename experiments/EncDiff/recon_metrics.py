import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import make_grid
from tqdm import tqdm


START_CWD = Path.cwd()
ENCDIFF_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ENCDIFF_DIR / "logs/2026-04-15T11-25-55_mpi3d-vq-4-16-encdiff-ct-v55"


def _prepare_imports():
    sys.path.insert(0, str(ENCDIFF_DIR))
    os.chdir(ENCDIFF_DIR)


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path

    from_start = START_CWD / path
    if from_start.exists():
        return from_start.resolve()

    from_encdiff = ENCDIFF_DIR / path
    return from_encdiff.resolve()


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())


def calculate_ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)


def calculate_lpips(x, y, loss_fn):
    """x/y: [B, 3, H, W], torch.Tensor in [-1, 1]."""
    return loss_fn(x, y).view(-1)


def calculate_mse(img1, img2):
    return (img1 - img2).pow(2).mean(dim=[1, 2, 3])


def find_project_config(run_dir):
    cfgs = sorted((run_dir / "configs").glob("*-project.yaml"))
    if not cfgs:
        raise FileNotFoundError(f"No *-project.yaml found under {run_dir / 'configs'}")
    return cfgs[-1]


def get_dataloader(data, split):
    if split == "train":
        return data._train_dataloader()
    if split == "validation":
        return data._val_dataloader()
    raise ValueError(f"Unknown split: {split}")


def to_image_range(x):
    return ((x.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def save_grid(x, path, nrow=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(to_image_range(x), nrow=nrow)
    grid = grid.transpose(0, 1).transpose(1, 2).numpy()
    grid = (grid * 255).astype(np.uint8)
    Image.fromarray(grid).save(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small DDIM reconstruction/generation smoke test for an EncDiff checkpoint."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory containing configs/ and checkpoints/.")
    parser.add_argument("--config", default=None, help="Optional project YAML. Defaults to <run-dir>/configs/*-project.yaml.")
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint. Defaults to <run-dir>/checkpoints/last.ckpt.")
    parser.add_argument("--output-dir", default=None, help="Where to save grids/metrics. Defaults inside the run dir.")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for this smoke test.")
    parser.add_argument("--num-workers", type=int, default=0, help="Small one-batch run does not need data-loader workers.")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--ddim-steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-images", type=int, default=8, help="Number of images to include in saved grids.")
    parser.add_argument("--nrow", type=int, default=4, help="Images per row in the saved grids.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS if you only need image outputs.")
    return parser.parse_args()


def main():
    _prepare_imports()

    from ldm.util import instantiate_from_config

    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    config_path = resolve_path(args.config) if args.config else find_project_config(run_dir)
    ckpt_path = resolve_path(args.ckpt) if args.ckpt else run_dir / "checkpoints/last.ckpt"
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else run_dir / f"recon_metrics_1batch_ddim{args.ddim_steps}"
    )

    print(f"Loading config: {config_path}", flush=True)
    config = OmegaConf.load(config_path)
    config.model.params.sampling_timesteps = args.ddim_steps
    config.data.params.batch_size = args.batch_size
    config.data.params.num_workers = args.num_workers

    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(args.device)
    model.eval()

    lpips_fn = None
    if not args.no_lpips:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").to(args.device).eval()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    lpips_scores = []
    ssim_scores = []
    mse_scores = []
    last_paths = {}

    loader = get_dataloader(data, args.split)
    batches = itertools.islice(loader, args.max_batches)
    iterator = tqdm(batches, total=args.max_batches, desc=f"Recon smoke test ({args.split})")
    for batch_idx, batch in enumerate(iterator):
        batch_image = batch["image"].to(args.device)
        batch_size = batch_image.shape[0]
        with torch.no_grad():
            z, cond, x, _x_vq, _xc = model.get_input(
                {"image": batch_image},
                model.first_stage_key,
                return_first_stage_outputs=True,
                force_c_encode=True,
                return_original_cond=True,
                bs=batch_size,
            )
            samples, _ = model.sample_log(
                cond=cond,
                batch_size=batch_size,
                ddim=True,
                ddim_steps=args.ddim_steps,
                eta=args.eta,
            )
            x_sample = model.decode_first_stage(samples)

        x_01 = to_image_range(x).to(args.device)
        sample_01 = to_image_range(x_sample).to(args.device)
        ssim_scores.append(calculate_ssim(x_01, sample_01).mean().item())
        mse_scores.append(calculate_mse(x_01, sample_01).mean().item())
        if lpips_fn is not None:
            lpips_scores.append(calculate_lpips(x, x_sample, lpips_fn).mean().item())

        n = min(args.max_images, batch_size)
        input_path = output_dir / f"batch{batch_idx:03d}_originals.png"
        recon_path = output_dir / f"batch{batch_idx:03d}_recons_ddim{args.ddim_steps}.png"

        save_grid(x[:n], input_path, nrow=args.nrow)
        save_grid(x_sample[:n], recon_path, nrow=args.nrow)

        last_paths = {
            "originals": str(input_path),
            "recons": str(recon_path),
        }

    summary = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "batch_size": args.batch_size,
        "num_batches": min(args.max_batches, len(lpips_scores) if lpips_scores else len(mse_scores)),
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "lpips": float(np.mean(lpips_scores)) if lpips_scores else None,
        "ssim": float(np.mean(ssim_scores)) if ssim_scores else None,
        "mse": float(np.mean(mse_scores)) if mse_scores else None,
        "outputs": last_paths,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved recon smoke-test outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
