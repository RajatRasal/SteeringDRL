from spatial_disentanglement.nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 1
    max_epochs = 500
    save_interval = 2
    eval_interval = 5
    save_epoch_end = False
    n_samples = 10  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 4e-4
    weight_decay = 0.0
    clip_grad = -1  # SA doesn't use any clipping
    warmup_steps_pct = 0.025  # SA warmup 10k steps over 500k total steps

    # data settings
    dataset = 'voc'
    data_root = './data/VOC'
    load_anno = True
    norm_mean = 0.5
    norm_std = 0.5
    train_batch_size = 64 // gpus
    val_batch_size = train_batch_size
    num_workers = 8
    grad_accum_steps = 1  # LDM saves 4x memory

    # model configs
    model = 'SA'
    resolution = (224, 224)

    # Slot Attention
    slot_size = 192
    slot_dict = dict(
        # DINOSAUR uses 6 slots on VOC
        num_slots=6,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=3,
    )

    # DINO encoder
    enc_dict = dict(
        dino='dino-vits8',
        enc_out_channels=slot_size,
        patch_size=8,  # downsample img by 8 or 16
        # True uses small sized dino (channel = 384)
        # False uses large sized dino (channel = 768)
        small_size=True,
        resolution=resolution,
    )

    # CNN Decoder
    dec_dict = dict(
        dec_channels=(slot_size, 128, 128, 128, 128),
        dec_resolution=(8, 8),
        dec_ks=5,
        dec_norm='',
    )

    # loss configs
    loss_dict = dict(use_img_recon_loss=True, )

    img_recon_loss_w = 1.  # slots image reconstruction loss weight
