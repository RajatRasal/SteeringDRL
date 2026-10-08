import argparse
import torch
from data.ground_truth import named_data
import numpy as np
import os
import json
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

import k_diffusion as K
from vqvae.VQVAE import VQVAE
from vqvae.losses.perceptual import VQLPIPSLoss
from vqvae.losses.perceptual_disc import VQLPIPSWithDiscriminatorLoss


def save_state_dict(net, path):
    """Save CPU model weights without overwriting an existing checkpoint."""
    state = net.state_dict()
    for name, tensor in state.items():
        state[name] = tensor.detach().cpu()
    with open(path, 'xb') as f:
        torch.save(state, f)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train a VQ-VAE for Shapes3D, Cars3D, or MPI3D.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--out-dir', required=True, help='Where to save the results')
    parser.add_argument('--dataset', choices=['shapes3d', 'mpi3d', 'cars3d'],
                        default='shapes3d', help='Training dataset')
    parser.add_argument('--disc', action='store_true', help='Include discriminator in loss')
    parser.add_argument('--duration', type=float, default=200, help='Training duration in MIMG')
    parser.add_argument('--batch', type=int, default=128, help='Total batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--tick', type=int, default=50, help='Progress reporting interval in KIMG')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--dump', type=int, default=50, help='Checkpoint interval in ticks')
    opts = parser.parse_args(argv)
    for name in ('duration', 'batch', 'lr', 'tick', 'dump'):
        if not getattr(opts, name) > 0:
            parser.error(f'--{name} must be greater than zero')
    if opts.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    return opts


def main(argv=None):
    opts = parse_args(argv)

    # Initialize.
    np.random.seed(opts.seed)
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    device = torch.device('cuda')

    opts.kimg_per_tick = opts.tick

    # Logging directory
    print("Creating output directory")
    os.makedirs(opts.out_dir, exist_ok=True)
    run_dir = os.path.join(opts.out_dir, f"{opts.seed}_{opts.dataset}_{opts.disc}_32_{opts.batch}_{opts.lr}")
    assert not os.path.exists(run_dir)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
        json.dump(vars(opts), f, indent=2)
    loss_columns = ['loss_perceptual', 'loss_discriminator'] if opts.disc else ['loss']
    metrics_log = K.utils.CSVLogger(
        os.path.join(run_dir, 'metrics.csv'), ['tick', 'nimgs', *loss_columns],
    )

    # Construct network
    print("Constructing network")
    net = VQVAE(
        enc_dec_dict=dict(
            resolution=64,
            in_channels=3,
            z_channels=3,
            ch=32,
            ch_mult=[1, 2, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            out_ch=3,
            dropout=0.0,
        ),
        vq_dict=dict(n_embed=2048, embed_dim=3),
    )
    net.train().requires_grad_(True).to(device)

    # Construct loss
    print("Constructing Loss")
    loss_fn = (VQLPIPSWithDiscriminatorLoss() if opts.disc else VQLPIPSLoss()).to(device)

    # Setup optimiser
    print("Setting up optimiser")
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=opts.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    if opts.disc:
        optimiser_disc = torch.optim.Adam(
            loss_fn.discriminator.parameters(),
            lr=opts.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    # Load dataset
    print("Loading dataset")
    dataset_obj = named_data.get_named_ground_truth_data(opts.dataset)
    _, eval_imgs = dataset_obj.sample(10000, np.random.RandomState(0))
    dataset_obj_eval = TensorDataset(torch.tensor(eval_imgs))
    dl_eval = DataLoader(
        dataset=dataset_obj_eval,
        batch_size=opts.batch,
        num_workers=opts.num_workers,
        shuffle=False,
        drop_last=True,
    )

    # Train
    total_kimg = max(int(opts.duration * 1000), 1)
    print(f"Training for {total_kimg} kimg...")
    print()
    state_dump_ticks = opts.dump
    cur_tick = 0
    cur_nimg = 0
    tick_start_nimg = cur_nimg
    loss_sums = [0.0] * len(loss_columns)
    loss_count = 0

    while True:
        # TODO: warmup = 0.05 of total steps
        # Training step
        _, images = dataset_obj.sample(opts.batch, np.random.RandomState(cur_nimg))
        images = torch.tensor(images).to(device).to(torch.float32).permute(0, 3, 1, 2) * 2 - 1

        if not opts.disc:
            optimiser.zero_grad()

            out_dict = net(images)
            loss, _ = loss_fn(out_dict["quant_loss"], images, out_dict["recon"])

            loss_sums[0] += loss.detach().item()

            loss.backward()
            optimiser.step()
        else:
            optimiser.zero_grad()

            out_dict = net(images)
            loss_p, _ = loss_fn(
                out_dict["quant_loss"],
                images,
                out_dict["recon"],
                optimizer_idx=0,
                global_step=cur_nimg,
                last_layer=net.decoder.conv_out.weight,
                cond=None,
                split="train",
            )
            loss_sums[0] += loss_p.detach().item()

            loss_p.backward()
            optimiser.step()

            optimiser_disc.zero_grad()

            out_dict = net(images)
            loss_d, _ = loss_fn(
                out_dict["quant_loss"],
                images,
                out_dict["recon"],
                optimizer_idx=1,
                global_step=cur_nimg,
                last_layer=net.decoder.conv_out.weight,
                cond=None,
                split="train",
            )
            loss_sums[1] += loss_d.detach().item()

            loss_d.backward()
            optimiser_disc.step()

        loss_count += 1

        # Perform maintenance tasks once per tick.
        cur_nimg += opts.batch
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + opts.kimg_per_tick * 1000):
            continue

        mean_losses = [value / loss_count for value in loss_sums]
        metrics_log.write(cur_tick, cur_nimg, *mean_losses)
        fields = [f'tick {cur_tick}', f'nimgs {cur_nimg}']
        fields.extend(f'{name} {value:.6g}' for name, value in zip(loss_columns, mean_losses))
        print(' '.join(fields))
        loss_sums = [0.0] * len(loss_columns)
        loss_count = 0

        # Save model weights in the format consumed by config.make_vqvae().
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0:
            save_state_dict(net, os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pth'))
            net.eval()
            # with torch.no_grad():
            eval_metrics = {} # "quant_loss": 0, "recon_loss": 0, "percept_loss": 0}
            for idx, images in enumerate(dl_eval):
                images = images[0].to(device).to(torch.float32).permute(0, 3, 1, 2) * 2 - 1
                out_dict = net(images)
                if opts.disc:
                    _, batch_metrics = loss_fn(
                        out_dict["quant_loss"],
                        images,
                        out_dict["recon"],
                        optimizer_idx=0,
                        global_step=cur_nimg,
                        last_layer=net.decoder.conv_out.weight,
                        cond=None,
                        split="val",
                    )
                else:
                    _, batch_metrics = loss_fn(out_dict["quant_loss"], images, out_dict["recon"])
                for k, v in batch_metrics.items():
                    if k not in eval_metrics:
                        eval_metrics[k] = v
                    else:
                        eval_metrics[k] += v
                if idx == 0:
                    recon = net.quantize_decode(net.encode(images)).clip(-1, 1)
                    recon = (recon + 1) * 0.5
                    images = (images + 1) * 0.5
                    imgs_grid = make_grid(images[:64], nrow=8)
                    recon_grid = make_grid(recon[:64], nrow=8)
                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].imshow(imgs_grid.permute(1, 2, 0).cpu().numpy())
                    axes[0].set_title("Original Images")
                    axes[0].axis("off")
                    axes[1].imshow(recon_grid.permute(1, 2, 0).cpu().numpy())
                    axes[1].set_title("Reconstructed Images")
                    axes[1].axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, f'training-state-recon-{cur_nimg//1000:06d}.pdf'))
                    plt.close(fig)
            eval_metrics = {k: v.item() / len(dl_eval) for k, v in eval_metrics.items()}
            net.train()
            with open(os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.json'), 'w') as f:
                json.dump(eval_metrics, f, indent=2)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        if done:
            print('Done!')
            break

    metrics_log.file.close()


if __name__ == "__main__":
    main()
