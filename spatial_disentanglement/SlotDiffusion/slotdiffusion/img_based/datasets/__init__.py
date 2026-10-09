from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.datasets.coco import build_coco_dataset, draw_coco_bbox
from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.datasets.voc import build_voc_dataset
from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.datasets.celeba import build_celeba_dataset
from spatial_disentanglement.SlotDiffusion.slotdiffusion.img_based.datasets.clevrtex import build_clevrtex_dataset


def build_dataset(params, val_only=False):
    dst = params['dataset']
    return eval(f'build_{dst}_dataset')(params, val_only=val_only)
