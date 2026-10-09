from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.models.slate.dVAE import dVAE
from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.models.slate.slate import SLATE

from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.models.slate.slate_utils import cosine_anneal, gumbel_softmax, make_one_hot


def build_model(params):
    """Build model."""
    if params.model == 'dVAE':
        model = dVAE(
            vocab_size=params.vocab_size,
            img_channels=3,
        )
    elif params.model == 'SLATE':
        model = SLATE(
            resolution=params.resolution,
            slot_dict=params.slot_dict,
            dvae_dict=params.dvae_dict,
            enc_dict=params.enc_dict,
            dec_dict=params.dec_dict,
            loss_dict=params.loss_dict,
        )
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
    return model
