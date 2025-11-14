import time
import dnnlib
import click
import torch
from data.ground_truth import named_data
import numpy as np
from torch_utils import training_stats
import pickle 
import psutil
import copy
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from torch_utils import training_stats
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


@click.command()
# Main options.
@click.option('--out-dir',       help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--dataset',       help='Path to the dataset', metavar='shapes3d|mpi3d|cars3d',       type=click.Choice(['shapes3d', 'mpi3d', 'cars3d']), default='shapes3d', show_default=True)
# Hyperparameters
@click.option('--disc',          help='Include discriminator in loss', metavar='BOOL',              type=bool, default=False)
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int, default=64)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--num-workers',          help='Dataloader workers', metavar='INT',                   type=int, default=0)
# I/O-related.
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int, default=0)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)

    # Initialize.
    start_time = time.time()
    np.random.seed(opts.seed)
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    device = torch.device('cuda')

    opts.kimg_per_tick = opts.tick
    
    opts.network_kwargs = dnnlib.EasyDict(
        class_name="vqvae.VQVAE.VQVAE",
        enc_dec_dict={
            "resolution": 64,
            "in_channels": 3,
            "z_channels": 3,
            "ch": opts.cbase,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "out_ch": 3,
            "dropout": 0.0,
        },
        vq_dict={
            "n_embed": 2048,
            "embed_dim": 3,
            # "percept_loss_w": 1.0,
        }
    )
    opts.loss_kwargs = dnnlib.EasyDict(
        class_name=f"vqvae.losses.{'perceptual_disc.VQLPIPSWithDiscriminatorLoss' if opts.disc else 'perceptual.VQLPIPSLoss'}",
        # percept_loss_w=1.0,
    )
    opts.optimiser_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam",
        lr=opts.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
    )

    # Logging directory
    print("Creating output directory")
    os.makedirs(opts.out_dir, exist_ok=True)
    run_dir = os.path.join(opts.out_dir, f"{opts.seed}_{opts.dataset}_{opts.disc}_{opts.cbase}_{opts.batch}_{opts.lr}")
    assert not os.path.exists(run_dir)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
        json.dump(opts, f, indent=2)
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Construct network
    print("Constructing network")
    net = dnnlib.util.construct_class_by_name(**opts.network_kwargs)
    net.train().requires_grad_(True).to(device)

    # Construct loss
    print("Constructing Loss")
    loss_fn = dnnlib.util.construct_class_by_name(**opts.loss_kwargs).to(device)

    # Setup optimiser
    print("Setting up optimiser")
    optimiser = dnnlib.util.construct_class_by_name(
        params=list(net.parameters()),
        **opts.optimiser_kwargs,
    )
    if opts.disc:
        optimiser_disc = dnnlib.util.construct_class_by_name(
            params=list(loss_fn.discriminator.parameters()),
            # TODO: Warmup with disc_start parameter
            **opts.optimiser_kwargs,
        )

    # Load dataset
    print("Loading dataset")
    dataset_obj = named_data.get_named_ground_truth_data(opts.dataset)
    _, eval_imgs = dataset_obj.sample(10000, np.random.RandomState(0))
    dataset_obj_eval = TensorDataset(torch.tensor(eval_imgs))
    dl_eval = DataLoader(
        dataset=dataset_obj_eval,
        batch_size=opts.batch,
        num_workers=0,
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
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None

    while True:
        # TODO: warmup = 0.05 of total steps
        # Training step
        _, images = dataset_obj.sample(opts.batch, np.random.RandomState(cur_nimg))
        images = torch.tensor(images).to(device).to(torch.float32).permute(0, 3, 1, 2) * 2 - 1

        if not opts.disc:
            optimiser.zero_grad()

            out_dict = net(images)
            loss, _ = loss_fn(out_dict["quant_loss"], images, out_dict["recon"])

            training_stats.report("Loss", loss)
            print(f"{cur_nimg}, loss:", loss.item())

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
            training_stats.report("Loss Perceptual", loss_p)

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
            training_stats.report("Loss Discriminator", loss_d)

            loss_d.backward()
            optimiser_disc.step()
            
            print(f"{cur_nimg}, P loss: {loss_p.item()}, D loss: {loss_d.item()}")

        # Perform maintenance tasks once per tick.
        cur_nimg += opts.batch
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + opts.kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        print(' '.join(fields))

        # Save network snapshot.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0:
            data = dict(optimizer_state=optimiser.state_dict(), step=cur_nimg, net=net)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    data[key] = value.cpu()
                del value # conserve memory
            with open(os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                pickle.dump(data, f)
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
            del data # conserve memory

        # Update logs.
        training_stats.default_collector.update()
        if stats_jsonl is None:
            stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
        stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
        stats_jsonl.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            print('Aborting...')
            break


if __name__ == "__main__":
    main()