from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.unet.image_v5 import UNetModelV5
from spatial_disentanglement.SlotDiffusion.slotdiffusion.video_based.models.unet.unet import UNetModel


def build_unet(config):
    config = dict(config)
    model_type = config.pop('type', 'original')

    if model_type in ['original', 'unet', 'slotdiffusion']:
        return UNetModel(**config)
    if model_type in ['image_v5', 'unet_v5']:
        return UNetModelV5(**config)
    raise ValueError(f'Unknown UNet type: {model_type}')


__all__ = ['UNetModel', 'UNetModelV5', 'build_unet']
