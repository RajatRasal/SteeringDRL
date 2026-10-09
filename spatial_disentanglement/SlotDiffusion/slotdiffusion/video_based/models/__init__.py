from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.savi import SAVi
from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.savi_diffusion import SAViDiffusion
from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.steve import build_model as build_steve_model
from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.vqvae import VQVAE, VQVAEWrapper

from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.utils import get_lr, to_rgb_from_tensor
from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.eval_utils import ARI_metric, fARI_metric, miou_metric, fmiou_metric, mbo_metric, mse_metric, psnr_metric, ssim_metric, perceptual_dist
from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.steve import cosine_anneal, gumbel_softmax, make_one_hot


def build_model(params):
    """Build model."""
    if params.model in ['SAVi', 'SAViDiffusion']:
        model = eval(params.model)(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            enc_dict=params.enc_dict,
            dec_dict=params.dec_dict,
            pred_dict=params.pred_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model in ['dVAE', 'STEVE']:
        model = build_steve_model(params)
    elif params.model == 'VQVAE':
        model = VQVAE(
            enc_dec_dict=params.enc_dec_dict,
            vq_dict=params.vq_dict,
        )
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
    return model
