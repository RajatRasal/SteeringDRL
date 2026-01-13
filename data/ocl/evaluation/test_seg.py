import os
import sys
import importlib
import argparse

import numpy as np
import torch

from data.ocl.evaluation.eval_utils import (
    ARI_metric,
    fARI_metric,
    miou_metric,
    fmiou_metric,
    mbo_metric,
)


@torch.no_grad()
def eval_masks(pred_masks, gt_masks, inst_overlap_masks=None):
    """Calculate seg_mask related metrics.

    Args:
        pred_masks: [B, N, H, W], argmaxed predicted masks.
        gt_masks: [B, H, W], argmaxed GT
    """
    pred_masks = pred_masks.argmax(dim=-3)  # [B, N, H, W] --> [B, H, W]
    gt_masks = gt_masks.long()
    # in the paper, we report FG-ARI, mIoU and mBO
    # we also provide other metrics to evaluate
    metrics = ['ARI', 'fARI', 'miou', 'fmiou', 'mbo']
    ret_dict = {
        m: eval(f'{m}_metric')(gt_masks, pred_masks, inst_overlap_masks)
        for m in metrics
    }
    return ret_dict


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, device=torch.device('cpu')):
        self.reset()
        self.device = device

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        if np.isnan(val):
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return torch.tensor(self.avg).to(self.device)

    def to(self, device):
        self.device = device
        return self



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test image segmentation')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--bs', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotAttentionParams()

    if args.bs > 0:
        params.val_batch_size = args.bs

    torch.backends.cudnn.benchmark = True
    main(params)
