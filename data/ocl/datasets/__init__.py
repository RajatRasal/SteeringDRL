from data.ocl.datasets.coco import build_coco_dataset, draw_coco_bbox
from data.ocl.datasets.voc import build_voc_dataset
from data.ocl.datasets.celeba import build_celeba_dataset
from data.ocl.datasets.clevrtex import build_clevrtex_dataset


def build_dataset(params, val_only=False):
    dst = params['dataset']
    return eval(f'build_{dst}_dataset')(params, val_only=val_only)
