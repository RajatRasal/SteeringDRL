from spatial_disentanglement.nerv.training import BaseParams


class SlotAttentionParams(BaseParams):
    project = 'SlotDiffusion'

    # training settings
    gpus = 1  # A40
    max_epochs = 500
    save_interval = 0.5
    eval_interval = 4
    save_epoch_end = True
    n_samples = 10  # visualization after each epoch

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4  # 1e-4 for the main SLATE model
    dec_lr = 3e-4  # 3e-4 for the Transformer decoder
    weight_decay = 0.0
    clip_grad = 1.  # following the paper
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

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
    model = 'SLATE'
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

    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='./pretrained/dvae_voc_params-res224.pth',
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

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=8,
        dec_num_heads=4,
        dec_d_model=slot_size,
    )

    # loss settings
    loss_dict = dict(
        use_img_recon_loss=False,  # additional img recon loss via dVAE decoder
    )

    token_recon_loss_w = 1.
    img_recon_loss_w = 1.
