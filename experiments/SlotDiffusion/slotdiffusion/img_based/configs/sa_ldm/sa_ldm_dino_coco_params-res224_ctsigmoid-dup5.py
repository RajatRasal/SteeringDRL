from nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 1
    max_epochs = 100
    save_interval = 0.25
    eval_interval = 4
    save_epoch_end = False
    n_samples = 10

    # optimizer settings
    optimizer = 'Adam'
    lr = 1e-4
    dec_lr = 2 * lr
    weight_decay = 0.0
    clip_grad = 0.05
    warmup_steps_pct = 0.05

    # data settings
    dataset = 'coco'
    data_root = './data/COCO'
    load_anno = True
    norm_mean = 0.5
    norm_std = 0.5
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size
    num_workers = 8
    grad_accum_steps = 1

    # model configs
    model = 'SADiffusion'
    resolution = (224, 224)
    img_ch = 3

    # Slot Attention
    slot_size = 256
    slot_dict = dict(
        num_slots=7,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=3,
    )

    # DINO encoder
    enc_dict = dict(
        dino='dino-vits8',
        enc_out_channels=slot_size,
        patch_size=8,
        small_size=True,
        resolution=resolution,
    )

    # LDM Decoder
    latent_ch = 3
    vae_dict = dict(
        vae_type='VQVAE',
        enc_dec_dict=dict(
            resolution=resolution[0],
            in_channels=img_ch,
            z_channels=latent_ch,
            ch=64,
            ch_mult=[1, 2, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            out_ch=img_ch,
            dropout=0.0,
        ),
        vq_dict=dict(
            n_embed=4096,
            embed_dim=latent_ch,
            percept_loss_w=1.0,
        ),
        vqvae_ckp_path='./pretrained/vqvae_coco_params-res224.pth',
    )

    unet_dict = dict(
        in_channels=latent_ch,
        model_channels=128,
        out_channels=latent_ch,
        num_res_blocks=2,
        attention_resolutions=(8, 4, 2),
        dropout=0.1,
        channel_mult=(1, 2, 3, 4),
        dims=2,
        use_checkpoint=False,
        num_head_channels=32,
        resblock_updown=False,
        conv_resample=True,
        transformer_depth=1,
        context_dim=slot_size,
        n_embed=None,
    )
    dec_dict = dict(
        resolution=tuple(res // 4 for res in resolution),
        vae_dict=vae_dict,
        unet_dict=unet_dict,
        use_ema=False,
        diffusion_dict=dict(
            pred_target='x0',
            z_scale_factor=1.,
            timesteps=1000,
            beta_schedule="linear",
            linear_start=0.0015,
            linear_end=0.0195,
            cosine_s=8e-3,
            log_every_t=200,
            logvar_init=0.,
            continuous_time=True,
            continuous_schedule='cosine',
            continuous_loss='sigmoid',
            logsnr_min=-15.0,
            logsnr_max=15.0,
            sigmoid_bias=-3.0,
            sampling_timesteps=200,
            continuous_dpm_fallback_steps=50,
        ),
        conditioning_key='crossattn',
        cond_stage_key='slots',
    )

    # loss configs
    loss_dict = dict(use_denoise_loss=True, )

    denoise_loss_w = 1.
