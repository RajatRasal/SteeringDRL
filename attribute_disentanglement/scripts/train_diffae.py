#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
import math
import json
from pathlib import Path
import time
import shutil
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, List

import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import multiprocessing as mp
from torch import optim
from torchvision import utils
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, StackDataset, TensorDataset
import matplotlib.pyplot as plt
import lpips
from torch import nn

import attribute_disentanglement.diffusion as K
from attribute_disentanglement.datasets import named_data


class GradNormLogger:
    """
    Tracks gradient norms with:
      - EMA (persistent across training)
      - Window stats (reset after each flush)
    
    And produces a ready-to-log dictionary via make_log_dict().
    """

    def __init__(self, ema_alpha: float = 0.9):
        assert 0.0 < ema_alpha < 1.0
        self.ema_alpha = ema_alpha

        # Persistent EMA values
        self._ema: Dict[str, float] = {}

        # Window of recent raw values (cleared after flush)
        self._window: Dict[str, List[float]] = defaultdict(list)

    @staticmethod
    def grad_norm(obj) -> float:
        """
        Compute L2 gradient norm.

        Supports:
          - nn.Module
          - Iterable of parameters (e.g. list[Parameter])

        Returns:
          float: sqrt(sum(g^2)).
        """
        total = 0.0

        # Case 1: nn.Module → use .parameters()
        if isinstance(obj, nn.Module):
            params = obj.parameters()

        # Case 2: list/tuple/set of parameters
        elif isinstance(obj, (list, tuple, set)):
            params = obj

        else:
            raise TypeError(
                f"grad_norm expected nn.Module or list of parameters, got {type(obj)}"
            )

        # accumulate squared grad norms
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += g.pow(2).sum().item()

        return math.sqrt(total + 1e-12)

    def update(self, values: Dict[str, float]):
        """
        Update EMA and window buffers with new gradient norms.

        Args:
            values: dict[name] = float
        """
        alpha = self.ema_alpha
        for name, v in values.items():
            v = float(v)

            # EMA update
            if name not in self._ema:
                self._ema[name] = v
            else:
                self._ema[name] = alpha * self._ema[name] + (1 - alpha) * v

            # add to window buffer
            self._window[name].append(v)

    def _compute_window_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute stats for current window (mean/std/min/max/count)."""
        stats = {}
        for name, vals in self._window.items():
            if not vals:
                continue
            n = len(vals)
            mean = sum(vals) / n
            var = sum((x - mean) ** 2 for x in vals) / max(1, n - 1)
            std = math.sqrt(var)
            stats[name] = {
                "mean": mean,
                "std": std,
                "min": min(vals),
                "max": max(vals),
                "count": n,
            }
        return stats

    def make_log_dict(self, prefix: str = "grads/", step: Optional[int] = None) -> Dict[str, Any]:
        """
        Produce a dictionary suitable for wandb/TensorBoard logging.

        Returns a dict of:
            prefix + name + "_ema"
            prefix + name + "_window_mean"
            prefix + name + "_window_std"
            prefix + name + "_window_min"
            prefix + name + "_window_max"
            prefix + name + "_window_count"

        Then clears window state.
        """
        log_dict = {}

        # --- window stats ---
        window_stats = self._compute_window_stats()
        for name, stats in window_stats.items():
            log_dict[f"{prefix}{name}_window_mean"] = stats["mean"]
            log_dict[f"{prefix}{name}_window_std"] = stats["std"]
            log_dict[f"{prefix}{name}_window_min"] = stats["min"]
            log_dict[f"{prefix}{name}_window_max"] = stats["max"]
            log_dict[f"{prefix}{name}_window_count"] = stats["count"]

        # --- ema values ---
        for name, ema in self._ema.items():
            log_dict[f"{prefix}{name}_ema"] = ema

        if step is not None:
            log_dict["step"] = step

        # clear window after flush
        self._window.clear()

        return log_dict


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=64,
                   help='the batch size')
    p.add_argument('--checkpointing', action='store_true',
                   help='enable gradient checkpointing')
    p.add_argument('--compile', action='store_true',
                   help='compile the model')
    p.add_argument('--config', type=str, required=True,
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--end-nimgs', type=int, default=None,
                   help='the nimgs to end training at')
    p.add_argument('--evaluate-every', type=int, default=10000,
                   help='evaluate every this many steps')
    p.add_argument('--debug-log-every', type=int, default=10000,
                   help='debug logging stats every this many steps')
    p.add_argument('--evaluate-only', action='store_true',
                   help='evaluate instead of training')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--gradient-clipping', action='store_true',
                   help='Clip gradients of encoder and diffusion decoder to 1.0')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
    p.add_argument('--exp-name', type=str, default='exp',
                   help='the name of the experiment group')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--reset-ema', action='store_true',
                   help='reset the EMA')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--resume-inference', type=str,
                   help='the inference checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--sampling-steps', type=int, default=50,
                   help='number of steps used when sampling')
    p.add_argument('--eta', type=float, default=1.0,
                   help='determinism of sampler from 0 = DDIM to 1 = DDPM')
    p.add_argument('--training-seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--wandb-entity', type=str,
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str,
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    args = p.parse_args()

    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch._dynamo.config.automatic_dynamic_shapes = False
    except AttributeError:
        pass
    device = torch.device("cuda")

    config = K.config.load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    encoder_config = config['encoder']

    # Set up run directory structure: <cwd>/ckpts/<name>/{checkpoints,images,configs}
    run_dir = Path(f'ckpts/{args.exp_name}') / args.name
    checkpoints_dir = run_dir / 'checkpoints'
    images_dir = run_dir / 'images'
    configs_dir = run_dir / 'configs'
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    images_dir.mkdir(parents=True, exist_ok=False)
    configs_dir.mkdir(parents=True, exist_ok=False)

    # Store a copy of the launch config file alongside the run
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        shutil.copy2(cfg_path, configs_dir / cfg_path.name)

    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    if args.seed is not None:
        torch.manual_seed(args.seed)
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    inner_model = K.config.make_model(config).to(device)
    inner_model_ema = deepcopy(inner_model)
    encoder = K.config.make_encoder(config).to(device)
    encoder_ema = deepcopy(encoder)
    vqvae = K.config.make_vqvae(config).to(device)
    vqvae.scale_factor = vqvae.scale_factor.to(device)

    if args.compile:
        inner_model.compile()
        encoder.compile()

    vqvae_n_params = K.utils.n_params(vqvae)
    unet_n_params = K.utils.n_params(inner_model)
    encoder_n_params = K.utils.n_params(encoder)
    total_n_params = vqvae_n_params + unet_n_params + encoder_n_params
    print(f'Total Parameters: {total_n_params:,}, VQVAE Parameters: {vqvae_n_params:,}, Encoder Parameters: {encoder_n_params:,}, UNet Parameters: {unet_n_params:,}')

    use_wandb = args.wandb_project
    if use_wandb:
        import wandb
        log_config = vars(args)
        log_config['config'] = config
        log_config['parameters'] = K.utils.n_params(inner_model)
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            config=log_config,
            save_code=True,
        )

    lr = opt_config['lr'] if args.lr is None else args.lr
    dec_lr = 2 * lr
    groups = inner_model.param_groups(dec_lr)
    groups.append({'params': encoder.parameters(), 'lr': lr})
    if opt_config['type'] == 'adamw':
        opt = optim.AdamW(
            groups,
            lr=lr,
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay'],
        )
    elif opt_config['type'] == 'adam8bit':
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(
            groups,
            lr=lr,
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay'],
        )
    elif opt_config['type'] == 'sgd':
        opt = optim.SGD(
            groups,
            lr=lr,
            momentum=opt_config.get('momentum', 0.),
            nesterov=opt_config.get('nesterov', False),
            weight_decay=opt_config.get('weight_decay', 0.),
        )
    else:
        raise ValueError('Invalid optimizer type')

    if sched_config['type'] == 'inverse':
        sched = K.utils.InverseLR(
            opt,
            inv_gamma=sched_config['inv_gamma'],
            power=sched_config['power'],
            warmup=sched_config['warmup'],
        )
    elif sched_config['type'] == 'exponential':
        sched = K.utils.ExponentialLR(
            opt,
            num_steps=sched_config['num_steps'],
            decay=sched_config['decay'],
            warmup=sched_config['warmup'],
        )
    elif sched_config['type'] == 'constant':
        sched = K.utils.ConstantLRWithWarmup(opt, warmup=sched_config['warmup'])
    elif sched_config['type'] == 'cosine':
        total_steps = args.end_nimgs // args.batch_size
        sched = K.utils.CosineAnnealingWarmupRestarts(
            opt,
            total_steps,
            max_lr=(dec_lr, dec_lr, lr),
            min_lr=0.,
            warmup_steps=int(total_steps * sched_config['warmup_pct']),
        )
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(
        power=ema_sched_config['power'],
        max_value=ema_sched_config['max_value'],
    )
    ema_stats = {}

    supported_datasets = {'shapes3d', 'cars3d', 'mpi3d'}
    if dataset_config['type'] not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset {dataset_config['type']!r}; "
            f"choose one of {sorted(supported_datasets)}"
        )
    dataset_obj = named_data.get_named_ground_truth_data(dataset_config['type'])

    with torch.no_grad(), K.models.flops.flop_counter() as fc:
        x = torch.zeros([1, model_config['input_channels'], size[0] * 4, size[1] * 4], device=device)
        if encoder_config['type'].startswith('image'):
            conds = encoder(x)
        elif encoder_config['type'] == 'label':
            labels = torch.zeros([1, dataset_obj.num_factors], device=device)
            conds = encoder(labels)
        else:
            raise NotImplementedError()
        x_latent = torch.zeros([1, model_config['input_channels'], size[0], size[1]], device=device)
        logsnr = torch.ones([1], device=device)
        extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
        out = inner_model(x_latent, logsnr, **extra_args)
        assert list(out.shape) == [1, model_config['input_channels'], size[0], size[1]]
        print(f'Forward pass GFLOPs: {fc.flops / 1_000_000_000:,.3f}', flush=True)

    if use_wandb:
        wandb.watch(inner_model)
    noise_schedule = K.config.make_lambda_noise_schedule(model_config)
    weighting_fn = K.config.make_weighting_function(model_config)

    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = run_dir / 'state.json'

    if state_path.exists() or args.resume:
        if args.resume:
            ckpt_path = args.resume
        if not args.resume:
            state = json.load(open(state_path))
            ckpt_path = state['latest_checkpoint']
        print(f'Resuming from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.inner_model.load_state_dict(ckpt['model'])
        model_ema.inner_model.load_state_dict(ckpt['model_ema'])
        encoder.load_state_dict(ckpt['encoder'])
        encoder_ema.load_state_dict(ckpt['encoder_ema'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        ema_sched.load_state_dict(ckpt['ema_sched'])
        ema_stats = ckpt.get('ema_stats', ema_stats)
        nimgs = ckpt['nimgs'] + 1
        demo_gen.set_state(ckpt['demo_gen'])
        elapsed = ckpt.get('elapsed', 0.0)
        del ckpt
    else:
        nimgs = 0

    if args.reset_ema:
        model.inner_model.load_state_dict(model_ema.inner_model.state_dict())
        ema_sched = K.utils.EMAWarmup(
            power=ema_sched_config['power'],
            max_value=ema_sched_config['max_value'],
        )
        ema_stats = {}

    if args.resume_inference:
        # if accelerator.is_main_process:
        print(f'Loading {args.resume_inference}...')
        ckpt = safetorch.load_file(args.resume_inference)
        model.inner_model.load_state_dict(ckpt)
        model_ema.inner_model.load_state_dict(ckpt)
        del ckpt

    evaluate_enabled = args.evaluate_every > 0
    metrics_log = None
    if evaluate_enabled:
        extractor = lpips.LPIPS(net='vgg').to(device).to(torch.float32)
        metrics = ['nimgs', 'time', 'loss', 'mse', 'lpips', 'dci', 'factorvae', 'mig', 'betavae']
        metrics_log = K.utils.CSVLogger(str(run_dir / 'metrics.csv'), metrics)

    debug_log_enabled = args.debug_log_every > 0
    debug_log = None
    if debug_log_enabled:
        debug_log = K.utils.CSVLogger(
            str(run_dir / 'debug.csv'),
            [
                'nimgs',
                'window_enc_mean',
                'window_enc_std',
                'window_enc_min',
                'window_enc_max',
                'window_dec_mean',
                'window_dec_std',
                'window_dec_min',
                'window_dec_max',
                'window_ca_mean',
                'window_ca_std',
                'window_ca_min',
                'window_ca_max',
                'ema_enc_mean',
                'ema_dec_mean',
                'ema_ca_mean',
            ],
        )

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    @K.utils.eval_mode(encoder_ema)
    def demo():
        tqdm.write('Sampling...')

        labels, images = dataset_obj.sample(args.sample_n, np.random.RandomState(0))
        images = K.utils.normalise(torch.tensor(images).to(device).to(torch.float32).permute(0, 3, 1, 2))
        labels = torch.tensor(labels).to(device).to(torch.float32)

        if encoder_config['type'].startswith('image'):
            conds = encoder_ema(images)
        elif encoder_config['type'] == 'label':
            conds = encoder_ema(labels)
        else:
            raise NotImplementedError()

        noise = torch.randn([args.sample_n, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
        extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
        recon = K.sampling.sample_ddim_lambda(model_ema, noise, noise_schedule, steps=args.sampling_steps, eta=args.eta, extra_args=extra_args)
        # recon = vqvae.quantize_decode(recon).clamp(-1, 1)
        recon = vqvae.decode(recon).clamp(-1, 1)

        recon_grid = utils.make_grid(
            K.utils.unnormalise(recon),
            nrow=math.ceil(args.sample_n ** 0.5),
            padding=0,
        )
        imgs_grid = utils.make_grid(
            K.utils.unnormalise(images),
            nrow=math.ceil(args.sample_n ** 0.5),
            padding=0,
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(imgs_grid.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Original Images')
        axes[0].axis('off')
        
        axes[1].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title('Reconstructed Images')
        axes[1].axis('off')

        plt.tight_layout()
        filename = str(images_dir / f'recon_grid_{nimgs:08}.png')
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

        if use_wandb:
            wandb.log({'recon_grid': wandb.Image(filename)}, nimg=nimgs)

        ######################

        tgt_image = images[-1].unsqueeze(0)
        if encoder_config['type'].startswith('image'):
            src_conds = encoder_ema.encoding(images)
            tgt_conds = encoder_ema.encoding(tgt_image)
        elif encoder_config['type'] == 'label':
            tgt_label = labels[-1].unsqueeze(0)
            src_conds = labels
            tgt_conds = tgt_label

        edit_conds_list = []
        for i in range(tgt_conds.shape[1]):
            edit_conds = src_conds.clone()
            edit_conds[:, i] = tgt_conds[0, i].clone()
            if encoder_config['type'].startswith('image'):
                edit_conds = encoder_ema.split_embedding(edit_conds)
            elif encoder_config['type'] == 'label':
                edit_conds = encoder_ema(edit_conds)
            edit_conds_list.append(edit_conds)

        edits = []
        for edit_conds in edit_conds_list:
            noise = torch.randn([args.sample_n, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
            extra_args = {'cross_cond': edit_conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
            edit = K.sampling.sample_ddim_lambda(model_ema, noise, noise_schedule, steps=args.sampling_steps, eta=args.eta, extra_args=extra_args)
            # edit = vqvae.quantize_decode(edit).clamp(-1, 1)
            edit = vqvae.decode(edit).clamp(-1, 1)
            edits.append(edit)
        summary_edits = [
            images,
            tgt_image.repeat(args.sample_n, 1, 1, 1),
        ]
        for edit in edits:
            summary_edits.append(edit.clip(-1, 1))
        summary_edits = K.utils.unnormalise(torch.concat(summary_edits))

        summary_edits_grid = utils.make_grid(
            summary_edits,
            nrow=args.sample_n,
            padding=0,
        )

        filename = str(images_dir / f'{args.name}_edits_grid_{nimgs:08}.png')
        plt.imshow(summary_edits_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        if use_wandb:
            wandb.log({'edit_grid': wandb.Image(filename)}, nimg=nimgs)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    @K.utils.eval_mode(encoder_ema)
    def evaluate():
        if not evaluate_enabled:
            return

        tqdm.write('Evaluating...')

        labels, images = dataset_obj.sample(10000, np.random.RandomState(0))
        images_ds = TensorDataset(torch.tensor(images).to(torch.float32))
        labels_ds = TensorDataset(torch.tensor(labels).to(torch.float32))
        ds = StackDataset(image=images_ds, label=labels_ds)
        dl = DataLoader(
            dataset=ds,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=True,
        )

        mse = 0
        lpips = 0

        for data in tqdm(dl, total=len(dl)):
            imgs = K.utils.normalise(data['image'][0].to(device).permute(0, 3, 1, 2))
            labels = data['label'][0].to(device)

            if encoder_config['type'].startswith('image'):
                conds = encoder_ema(imgs)
            elif encoder_config['type'] == 'label':
                conds = encoder_ema(labels)
            else:
                raise NotImplementedError()

            noise = torch.randn([args.batch_size, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device)
            extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
            recon = K.sampling.sample_ddim_lambda(model_ema, noise, noise_schedule, steps=args.sampling_steps, eta=args.eta, extra_args=extra_args)
            # recon = vqvae.quantize_decode(recon).clamp(-1, 1)
            recon = vqvae.decode(recon).clamp(-1, 1)

            mse_ = K.evaluation.mse_metric(
                K.utils.unnormalise(imgs),
                K.utils.unnormalise(recon),
            )
            lpips_ = K.evaluation.perceptual_dist(
                imgs,
                recon.to(torch.float32),
                extractor,
            )
            mse += mse_
            lpips += lpips_

        mse = mse / len(dl)
        lpips = lpips / len(dl)
        tqdm.write(f'MSE: {mse.item():g}, LPIPS: {lpips.item():g}')

        if encoder_config['type'].startswith('image'):
            dis_metrics = K.evaluation.evaluate_disentanglement(
                encoder_ema,
                dataset_obj,
                True,
                True,
                True,
                True,
                device,
            )
            dci = dis_metrics['DCI']['disentanglement']
            factorvae = dis_metrics['FactorVAE']['eval_accuracy']
            mig = dis_metrics['MIG']['discrete_mig']
            betavae = dis_metrics['BetaVAE']['eval_accuracy']
            dis_metrics = {
                "DCI": dci, "FactorVAE": factorvae, "MIG": mig, "BetaVAE": betavae
            }
        else:
            dis_metrics = {
                "DCI": 0,
                "FactorVAE": 0,
                "MIG": 0,
                "BetaVAE": 0,
            }

        metrics_log.write(
            nimgs,
            elapsed,
            ema_stats.get('loss', 0),
            mse.item(),
            lpips.item(),
            *dis_metrics.values(),
        )
        if use_wandb:
            wandb.log(
                {
                    'MSE': mse.item(),
                    'LPIPS': lpips.item(),
                    **dis_metrics,
                },
                step=nimgs,
            )

    def save():
        filename = str(checkpoints_dir / f'{args.name}_{nimgs:08}.pth')
        tqdm.write(f'Saving to {filename}...')
        inner_model = model.inner_model
        inner_model_ema = model_ema.inner_model
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'encoder': encoder.state_dict(),
            'encoder_ema': encoder_ema.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'nimgs': nimgs,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        torch.save(obj, filename)
        state_obj = {'latest_checkpoint': str(filename)}
        json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    if args.evaluate_only:
        if not evaluate_enabled:
            raise ValueError('--evaluate-only requested but evaluation is disabled')
        evaluate()
        return

    losses_since_last_print = []
    training_generator = torch.Generator().manual_seed(args.training_seed)

    def get_all_cross_attn_params(model) -> List:
        params = []
        for module in model.modules():
            if isinstance(module, K.layers.CrossAttention2d) or isinstance(module, K.layers.CrossAttention2dReZero):
                for p in module.parameters():
                    if p.requires_grad:
                        params.append(p)
        return params

    grad_logger = GradNormLogger()
    ca_module_params = get_all_cross_attn_params(model)

    try:
        while True:
            factors, reals = dataset_obj.sample(args.batch_size, np.random.RandomState(nimgs))
            reals = K.utils.normalise(torch.tensor(reals).to(device).to(torch.float32).permute(0, 3, 1, 2))
            factors = torch.tensor(factors).to(device).to(torch.float32)

            if device.type == 'cuda':
                start_timer = torch.cuda.Event(enable_timing=True)
                end_timer = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_timer.record()
            else:
                start_timer = time.time()

            with K.models.checkpointing(args.checkpointing):
                if encoder_config['type'].startswith('image'):
                    conds = encoder(reals)
                elif encoder_config['type'] == 'label':
                    conds = encoder(factors)
                else:
                    raise NotImplementedError()
                extra_args = {
                    'cross_cond': conds,
                    'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool),
                }

                with torch.no_grad():
                    reals = vqvae.encode(reals)

                losses = model.loss(
                    reals,
                    weighting_fn,
                    noise_schedule,
                    training_generator,
                    **extra_args,
                )

            loss = losses.mean()
            loss.backward()

            grad_values = {
                'encoder': GradNormLogger.grad_norm(encoder),
                'decoder': GradNormLogger.grad_norm(model),
                'ca': GradNormLogger.grad_norm(ca_module_params),
            }
            grad_logger.update(grad_values)

            if args.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)

            loss = loss.item()
            losses_since_last_print.append(loss)

            opt.step()
            sched.step()
            opt.zero_grad()

            ema_decay = ema_sched.get_value()
            K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
            K.utils.ema_update(model, model_ema, ema_decay)
            K.utils.ema_update(encoder, encoder_ema, ema_decay)
            ema_sched.step()

            if device.type == 'cuda':
                end_timer.record()
                torch.cuda.synchronize()
                elapsed += start_timer.elapsed_time(end_timer) / 1000
            else:
                elapsed += time.time() - start_timer

            if (nimgs // args.batch_size) % 25 == 0:
                loss_disp = sum(losses_since_last_print) / len(losses_since_last_print)
                losses_since_last_print.clear()
                avg_loss = ema_stats['loss']
                tqdm.write(f'step: {nimgs // args.batch_size}, nimgs: {nimgs}, loss: {loss_disp:g}, avg loss: {avg_loss:g}')

            if use_wandb:
                log_dict = {
                    'loss': loss,
                    'lr': sched.get_last_lr()[0],
                    'ema_decay': ema_decay,
                }
                wandb.log(log_dict, step=nimgs)

            nimgs += args.batch_size

            step = nimgs // args.batch_size

            if step == 50 or (step % args.demo_every == 0):
                demo()

            if debug_log_enabled and nimgs > 0 and step % args.debug_log_every == 0:
                log_dict = grad_logger.make_log_dict(step=nimgs)
                debug_log.write(
                    log_dict['step'],
                    log_dict['grads/encoder_window_mean'],
                    log_dict['grads/encoder_window_std'],
                    log_dict['grads/encoder_window_min'],
                    log_dict['grads/encoder_window_max'],
                    log_dict['grads/decoder_window_mean'],
                    log_dict['grads/decoder_window_std'],
                    log_dict['grads/decoder_window_min'],
                    log_dict['grads/decoder_window_max'],
                    log_dict['grads/ca_window_mean'],
                    log_dict['grads/ca_window_std'],
                    log_dict['grads/ca_window_min'],
                    log_dict['grads/ca_window_max'],
                    log_dict['grads/encoder_ema'],
                    log_dict['grads/decoder_ema'],
                    log_dict['grads/ca_ema'],
                )

            if evaluate_enabled and nimgs > 0:
                if step == 50 or (step % args.evaluate_every == 0):
                    evaluate()

            if nimgs >= args.end_nimgs or (nimgs > 0 and step % args.save_every == 0):
                save()

            if nimgs >= args.end_nimgs:
                tqdm.write('Done!')
                return

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
