#!/usr/bin/env python3

"""Trains Karras et al. (2022) diffusion models."""

import argparse
from copy import deepcopy
import math
import json
from pathlib import Path
import time
import numpy as np

import safetensors.torch as safetorch
import torch
import torch._dynamo
from torch import multiprocessing as mp
from torch import optim
from torch.utils import data, flop_counter
from torchvision import utils
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, StackDataset, TensorDataset
import matplotlib.pyplot as plt
import lpips

import k_diffusion as K
from data.ground_truth import named_data


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
    p.add_argument('--evaluate-only', action='store_true',
                   help='evaluate instead of training')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--mixed-precision', type=str,
                   help='the mixed precision type')
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

    config = K.config.load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    encoder_config = config['encoder']

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']
    size[0] = size[0] # int(size[0] / 4)
    size[1] = size[1] # int(size[1] / 4)

    device = torch.device("cuda")

    if args.seed is not None:
        torch.manual_seed(args.seed)
    demo_gen = torch.Generator().manual_seed(torch.randint(-2 ** 63, 2 ** 63 - 1, ()).item())
    elapsed = 0.0

    inner_model = K.config.make_model(config).to(device)
    inner_model_ema = deepcopy(inner_model)
    encoder = K.config.make_encoder(config).to(device)
    vqvae = K.config.make_vqvae(config).to(device)

    if args.compile:
        inner_model.compile()
        encoder.compile()

    print(f'Parameters: {K.utils.n_params(inner_model) + K.utils.n_params(encoder):,}')

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
    groups = inner_model.param_groups(lr)
    groups.append({"params": encoder.parameters(), "lr": lr})
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
    else:
        raise ValueError('Invalid schedule type')

    assert ema_sched_config['type'] == 'inverse'
    ema_sched = K.utils.EMAWarmup(
        power=ema_sched_config['power'],
        max_value=ema_sched_config['max_value'],
    )
    ema_stats = {}

    dataset_obj = named_data.get_named_ground_truth_data(dataset_config['type'])

    with torch.no_grad(), K.models.flops.flop_counter() as fc:
        x = torch.zeros([1, model_config['input_channels'], size[0] * 4, size[1] * 4], device=device)
        labels = torch.zeros([1, dataset_obj.num_factors], device=device)
        if encoder_config['type'] == 'image':
            conds = encoder(x)
        elif encoder_config['type'] == 'label':
            conds = encoder(labels)
        else:
            raise NotImplementedError()
        x_latent = torch.zeros([1, model_config['input_channels'], size[0], size[1]], device=device)
        sigma = torch.ones([1], device=device)
        extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
        inner_model(x_latent, sigma, **extra_args)
        print(f"Forward pass GFLOPs: {fc.flops / 1_000_000_000:,.3f}", flush=True)
    # exit()

    if use_wandb:
        wandb.watch(inner_model)
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    sample_density = K.config.make_sample_density(model_config)

    model = K.config.make_denoiser_wrapper(config)(inner_model)
    model_ema = K.config.make_denoiser_wrapper(config)(inner_model_ema)

    state_path = Path(f'{args.name}_state.json')

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
        metrics_log = K.utils.CSVLogger(
            f'{args.name}_metrics.csv',
            ['nimgs', 'time', 'loss', 'mse', 'lpips', 'dci', 'factorvae', 'mig', 'betavae'],
        )

    # cfg_scale = 1.

    # def make_cfg_model_fn(model):
    #     def cfg_model_fn(x, sigma, class_cond):
    #         x_in = torch.cat([x, x])
    #         sigma_in = torch.cat([sigma, sigma])
    #         class_uncond = torch.full_like(class_cond, num_classes)
    #         class_cond_in = torch.cat([class_uncond, class_cond])
    #         out = model(x_in, sigma_in, class_cond=class_cond_in)
    #         out_uncond, out_cond = out.chunk(2)
    #         return out_uncond + (out_cond - out_uncond) * cfg_scale
    #     if cfg_scale != 1:
    #         return cfg_model_fn
    #     return model

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    @K.utils.eval_mode(encoder)
    def demo():
        tqdm.write('Sampling...')

        s_noise = 1.0

        labels, images = dataset_obj.sample(args.sample_n, np.random.RandomState(0))
        images = K.utils.normalise(torch.tensor(images).to(torch.float32).to(device).permute(0, 3, 1, 2))
        labels = torch.tensor(labels).to(torch.float32).to(device)

        if encoder_config['type'] == 'image':
            conds = encoder(images)
        elif encoder_config['type'] == 'label':
            conds = encoder(labels)
        else:
            raise NotImplementedError()

        for rho in [1., 3., 5., 7., 9.]:
            sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=rho, device=device)
            x = torch.randn([args.sample_n, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device) * sigmas[0]
            extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
            # recon = K.sampling.sample_dpmpp_2m_sde(model_ema, x, sigmas, extra_args=extra_args, eta=1.0, solver_type='heun', disable=True)
            recon = K.sampling.sample_dpm_2(model_ema, x, sigmas, extra_args=extra_args, s_noise=s_noise, disable=True)
            recon = vqvae.quantize_decode(recon.clamp(-1, 1)).clamp(-1, 1)

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
            axes[0].set_title("Original Images")
            axes[0].axis("off")
        
            axes[1].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
            axes[1].set_title("Reconstructed Images")
            axes[1].axis("off")

            plt.tight_layout()
            filename = f'{args.name}_recon_grid_karras_rho{rho}_{nimgs:08}.png'
            plt.savefig(filename, bbox_inches="tight")
            plt.close(fig)

            if use_wandb:
                wandb.log({f'recon_grid_karras_rho{rho}': wandb.Image(filename)}, nimg=nimgs)

        sigmas = K.sampling.get_sigmas_vp(200, device=device)
        x = torch.randn([args.sample_n, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device) * sigmas[0]
        extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
        recon = K.sampling.sample_dpm_2(model_ema, x, sigmas, extra_args=extra_args, s_noise=s_noise, disable=True)
        recon = vqvae.quantize_decode(recon.clamp(-1, 1)).clamp(-1, 1)

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
        axes[0].set_title("Original Images")
        axes[0].axis("off")
        
        axes[1].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("Reconstructed Images")
        axes[1].axis("off")

        plt.tight_layout()
        filename = f'{args.name}_recon_grid_vp_{nimgs:08}.png'
        plt.savefig(filename, bbox_inches="tight")
        plt.close(fig)

        if use_wandb:
            wandb.log({f'recon_grid_vp': wandb.Image(filename)}, nimg=nimgs)

        ######################

        tgt_image = images[-1].unsqueeze(0)
        tgt_label = labels[-1].unsqueeze(0)
        if encoder_config['type'] == 'image':
            src_conds = encoder.encoding(images)
            tgt_conds = encoder.encoding(tgt_image)
        elif encoder_config['type'] == 'label':
            src_conds = labels
            tgt_conds = tgt_label

        edit_conds_list = []
        for i in range(tgt_conds.shape[1]):
            edit_conds = src_conds.clone()
            edit_conds[:, i] = tgt_conds[0, i].clone()
            if encoder_config['type'] == 'image':
                edit_conds = encoder.split_embedding(edit_conds)
            elif encoder_config['type'] == 'label':
                edit_conds = encoder(edit_conds)
            edit_conds_list.append(edit_conds)

        for rho in [1., 3., 5., 7., 9.]:
            edits = []
            for edit_conds in edit_conds_list:
                sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=rho, device=device)
                x = torch.randn([args.sample_n, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device) * sigmas[0]
                extra_args = {'cross_cond': edit_conds, 'cross_cond_padding': torch.zeros_like(edit_conds[..., 0], dtype=torch.bool)}
                edit = K.sampling.sample_dpm_2(model_ema, x, sigmas, extra_args=extra_args, s_noise=s_noise, disable=True)
                edits.append(vqvae.quantize_decode(edit.clamp(-1, 1)).clamp(-1, 1))
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

            filename = f'{args.name}_edits_grid_karras_rho{rho}_{nimgs:08}.png'
            plt.imshow(summary_edits_grid.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

            if use_wandb:
                wandb.log({f'edit_grid_karras_rho{rho}': wandb.Image(filename)}, nimg=nimgs)
        
        edits = []
        for edit_conds in edit_conds_list:
            sigmas = K.sampling.get_sigmas_vp(200, device=device)
            x = torch.randn([args.sample_n, model_config['input_channels'], size[0], size[1]], generator=demo_gen).to(device) * sigmas[0]
            extra_args = {'cross_cond': edit_conds, 'cross_cond_padding': torch.zeros_like(edit_conds[..., 0], dtype=torch.bool)}
            edit = K.sampling.sample_dpm_2(model_ema, x, sigmas, extra_args=extra_args, s_noise=s_noise, disable=True)
            edits.append(vqvae.quantize_decode(edit.clamp(-1, 1)).clamp(-1, 1))
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

        filename = f'{args.name}_edits_grid_vp_{nimgs:08}.png'
        plt.imshow(summary_edits_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        if use_wandb:
            wandb.log({f'edit_grid_vp': wandb.Image(filename)}, nimg=nimgs)

    @torch.no_grad()
    @K.utils.eval_mode(model_ema)
    @K.utils.eval_mode(encoder)
    def evaluate():
        if not evaluate_enabled:
            return

        tqdm.write('Evaluating...')
        s_noise = 1.0

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

        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)

        mse = 0
        lpips = 0

        for data in tqdm(dl, total=len(dl)):
            imgs = K.utils.normalise(data["image"][0].to(device).permute(0, 3, 1, 2))
            labels = data["label"][0].to(device)

            if encoder_config['type'] == 'image':
                conds = encoder(imgs)
            elif encoder_config['type'] == 'label':
                conds = encoder(labels)
            else:
                raise NotImplementedError()

            extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
            noise = torch.randn([args.batch_size, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            recon = K.sampling.sample_dpm_2(model_ema, noise, sigmas, extra_args=extra_args, s_noise=s_noise, disable=True)
            recon = vqvae.quantize_decode(recon.clamp(-1, 1)).clamp(-1, 1)

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

        # TODO: Add disentanglement evaluation
        if encoder_config['type'] == 'image':
            dis_metrics = K.evaluation.evaluate_disentanglement(
                encoder,
                dataset_obj,
                True,
                True,
                True,
                True,
                device,
            )
            dci = dis_metrics['dci_score']['disentanglement']
            factorvae = dis_metrics['factor_VAE_score']['eval_accuracy']
            mig = dis_metrics['MIG']['discrete_mig']
            betavae = dis_metrics['beta_vae_score']['eval_accuracy']
        else:
            dci = 0
            factorvae = 0
            mig = 0
            betavae = 0

        metrics_log.write(
            nimgs,
            elapsed,
            ema_stats['loss'],
            mse.item(),
            lpips.item(),
            dci,
            factorvae,
            mig,
            betavae,
        )
        if use_wandb:
            wandb.log(
                {
                    'MSE': mse.item(),
                    'LPIPS': lpips.item(),
                    'DCI': dci,
                    'FactorVAE': factorvae,
                    'MIG': mig,
                    'BetaVAE': betavae,
                },
                step=nimgs,
            )

    def save():
        filename = f'{args.name}_{nimgs:08}.pth'
        tqdm.write(f'Saving to {filename}...')
        inner_model = model.inner_model
        inner_model_ema = model_ema.inner_model
        obj = {
            'config': config,
            'model': inner_model.state_dict(),
            'model_ema': inner_model_ema.state_dict(),
            'encoder': encoder.state_dict(),
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
            'ema_sched': ema_sched.state_dict(),
            'nimgs': nimgs,
            'ema_stats': ema_stats,
            'demo_gen': demo_gen.get_state(),
            'elapsed': elapsed,
        }
        torch.save(obj, filename)
        state_obj = {'latest_checkpoint': filename}
        json.dump(state_obj, open(state_path, 'w'))
        if args.wandb_save_model and use_wandb:
            wandb.save(filename)

    if args.evaluate_only:
        if not evaluate_enabled:
            raise ValueError('--evaluate-only requested but evaluation is disabled')
        evaluate()
        return

    losses_since_last_print = []

    try:
        while True:
            factors, reals = dataset_obj.sample(args.batch_size, np.random.RandomState(nimgs))
            reals = K.utils.normalise(torch.tensor(reals).to(device).to(torch.float32).permute(0, 3, 1, 2))
            factors = torch.tensor(factors).to(device).to(torch.float32)

            factors = factors.to(device)
            reals = reals.to(device)

            if device.type == 'cuda':
                start_timer = torch.cuda.Event(enable_timing=True)
                end_timer = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_timer.record()
            else:
                start_timer = time.time()

            # print(reals.shape, reals.min(), reals.max())
            # if num_classes:
            #     class_cond = factors # batch[class_key]
            #     drop = torch.rand(class_cond.shape, device=class_cond.device)
            #     class_cond.masked_fill_(drop < cond_dropout_rate, num_classes)
            #     extra_args['class_cond'] = class_cond
            with K.models.checkpointing(args.checkpointing):
                if encoder_config['type'] == 'image':
                    conds = encoder(reals)
                elif encoder_config['type'] == 'label':
                    conds = encoder(factors)
                else:
                    raise NotImplementedError()
                extra_args = {'cross_cond': conds, 'cross_cond_padding': torch.zeros_like(conds[..., 0], dtype=torch.bool)}
                with torch.no_grad():
                    reals = vqvae.encode(reals)
                noise = torch.randn_like(reals)
                sigma = sample_density([reals.shape[0]], device=device)
                losses = model.loss(reals, noise, sigma, **extra_args)
            loss = losses.mean()
            loss.backward()
            loss = loss.item()
            losses_since_last_print.append(loss)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
            opt.step()
            sched.step()
            opt.zero_grad()
            # exit()

            ema_decay = ema_sched.get_value()
            K.utils.ema_update_dict(ema_stats, {'loss': loss}, ema_decay ** (1 / args.grad_accum_steps))
            K.utils.ema_update(model, model_ema, ema_decay)
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

            if (nimgs // args.batch_size) % args.demo_every == 0:
                demo()

            if evaluate_enabled and nimgs > 0 and (nimgs // args.batch_size) % args.evaluate_every == 0:
                evaluate()

            if (nimgs // args.batch_size) == args.end_nimgs or (nimgs > 0 and (nimgs // args.batch_size) % args.save_every == 0):
                save()

            if nimgs == args.end_nimgs:
                tqdm.write('Done!')
                return

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
