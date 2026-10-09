import math
import os
from pathlib import Path

from cleanfid.inception_torchscript import InceptionV3W
import clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm.auto import trange
import numpy as np

from attribute_disentanglement.diffusion import utils
import attribute_disentanglement.diffusion as K


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        path = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'k-diffusion'
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        digest = 'f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
        utils.download_file(path / 'inception-2015-12-05.pt', url, digest)
        self.model = InceptionV3W(str(path), resize_inside=False).to(device)
        self.size = (299, 299)

    def forward(self, x):
        x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = (x * 127.5 + 127.5).clamp(0, 255)
        return self.model(x)


class CLIPFeatureExtractor(nn.Module):
    def __init__(self, name='ViT-B/16', device='cpu'):
        super().__init__()
        self.model = clip.load(name, device=device)[0].eval().requires_grad_(False)
        self.normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                              std=(0.26862954, 0.26130258, 0.27577711))
        self.size = self.model.visual.input_resolution, self.model.visual.input_resolution

    @classmethod
    def available_models(cls):
        return clip.available_models()

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        x = self.model.encode_image(x).float()
        x = F.normalize(x) * x.shape[-1] ** 0.5
        return x


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, name='vitl14', device='cpu'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_' + name).to(device).eval().requires_grad_(False)
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.size = 224, 224

    @classmethod
    def available_models(cls):
        return ['vits14', 'vitb14', 'vitl14', 'vitg14']

    def forward(self, x):
        x = (x + 1) / 2
        x = F.interpolate(x, self.size, mode='bicubic', align_corners=False, antialias=True)
        if x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.normalize(x)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = self.model(x).float()
        x = F.normalize(x) * x.shape[-1] ** 0.5
        return x


def compute_features(accelerator, sample_fn, extractor_fn, n, batch_size):
    n_per_proc = math.ceil(n / accelerator.num_processes)
    feats_all = []
    try:
        for i in trange(0, n_per_proc, batch_size, disable=not accelerator.is_main_process):
            cur_batch_size = min(n - i, batch_size)
            samples = sample_fn(cur_batch_size)[:cur_batch_size]
            feats_all.append(accelerator.gather(extractor_fn(samples)))
    except StopIteration:
        pass
    return torch.cat(feats_all)[:n]


def polynomial_kernel(x, y):
    d = x.shape[-1]
    dot = x @ y.transpose(-2, -1)
    return (dot / d + 1) ** 3


def squared_mmd(x, y, kernel=polynomial_kernel):
    m = x.shape[-2]
    n = y.shape[-2]
    kxx = kernel(x, x)
    kyy = kernel(y, y)
    kxy = kernel(x, y)
    kxx_sum = kxx.sum([-1, -2]) - kxx.diagonal(dim1=-1, dim2=-2).sum(-1)
    kyy_sum = kyy.sum([-1, -2]) - kyy.diagonal(dim1=-1, dim2=-2).sum(-1)
    kxy_sum = kxy.sum([-1, -2])
    term_1 = kxx_sum / m / (m - 1)
    term_2 = kyy_sum / n / (n - 1)
    term_3 = kxy_sum * 2 / m / n
    return term_1 + term_2 - term_3


@utils.tf32_mode(matmul=False)
def kid(x, y, max_size=5000):
    x_size, y_size = x.shape[0], y.shape[0]
    n_partitions = math.ceil(max(x_size / max_size, y_size / max_size))
    total_mmd = x.new_zeros([])
    for i in range(n_partitions):
        cur_x = x[round(i * x_size / n_partitions):round((i + 1) * x_size / n_partitions)]
        cur_y = y[round(i * y_size / n_partitions):round((i + 1) * y_size / n_partitions)]
        total_mmd = total_mmd + squared_mmd(cur_x, cur_y)
    return total_mmd / n_partitions


class _MatrixSquareRootEig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        vals, vecs = torch.linalg.eigh(a)
        ctx.save_for_backward(vals, vecs)
        return vecs @ vals.abs().sqrt().diag_embed() @ vecs.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        vals, vecs = ctx.saved_tensors
        d = vals.abs().sqrt().unsqueeze(-1).repeat_interleave(vals.shape[-1], -1)
        vecs_t = vecs.transpose(-2, -1)
        return vecs @ (vecs_t @ grad_output @ vecs / (d + d.transpose(-2, -1))) @ vecs_t


def sqrtm_eig(a):
    if a.ndim < 2:
        raise RuntimeError('tensor of matrices must have at least 2 dimensions')
    if a.shape[-2] != a.shape[-1]:
        raise RuntimeError('tensor must be batches of square matrices')
    return _MatrixSquareRootEig.apply(a)


@utils.tf32_mode(matmul=False)
def fid(x, y, eps=1e-8):
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    mean_term = (x_mean - y_mean).pow(2).sum()
    x_cov = torch.cov(x.T)
    y_cov = torch.cov(y.T)
    eps_eye = torch.eye(x_cov.shape[0], device=x_cov.device, dtype=x_cov.dtype) * eps
    x_cov = x_cov + eps_eye
    y_cov = y_cov + eps_eye
    x_cov_sqrt = sqrtm_eig(x_cov)
    cov_term = torch.trace(x_cov + y_cov - 2 * sqrtm_eig(x_cov_sqrt @ y_cov @ x_cov_sqrt))
    return mean_term + cov_term


def mse_metric(x, y):
    """x/y: [B, 3, H, W], np.ndarray in [0, 1]"""
    # people often sum over channel + spatial dimension in recon MSE
    return ((x - y) ** 2).sum(-1).sum(-1).sum(-1).mean()


def perceptual_dist(x, y, loss_fn):
    """x/y: [B, 3, H, W], torch.Tensor in [-1, 1]"""
    return loss_fn(x, y).mean()


def evaluate_disentanglement(
    net,
    dataset=None,
    beta_VAE_score=False,
    dci_score=False,
    factor_VAE_score=False,
    MIG=False,
    device="cuda",
):
    def _representation(x):
        x = torch.tensor(x).to(device).to(torch.float32).permute(0, 3, 1, 2)
        x = K.utils.normalise(x)
        z = net.encoding(x)
        return z.detach().cpu().numpy()

    metrics = {}

    if beta_VAE_score:
        from attribute_disentanglement.metrics.beta_vae import compute_beta_vae_sklearn
        result_dict = compute_beta_vae_sklearn(
            dataset,
            _representation,
            random_state=np.random.RandomState(0),
            artifact_dir=None,
            num_train=10000,
            num_eval=5000,
            batch_size=64,
        )
        print("beta VAE score:" + str(result_dict))
        metrics["BetaVAE"] = result_dict
    if dci_score:
        from attribute_disentanglement.metrics.dci import compute_dci
        result_dict = compute_dci(
            dataset,
            _representation,
            random_state=np.random.RandomState(0),
            artifact_dir=None,
            num_train=10000,
            num_test=5000,
            batch_size=64,
        )
        print("dci score:" + str(result_dict))
        metrics["DCI"] = result_dict
    if factor_VAE_score:
        from attribute_disentanglement.metrics.factor_vae import compute_factor_vae
        result_dict = compute_factor_vae(
            dataset,
            _representation,
            random_state=np.random.RandomState(0),
            artifact_dir=None,
            batch_size=64,
            num_train=10000,
            num_eval=5000,
            num_variance_estimate=10000,
            prune_threshold=0.05,
        )
        print("factor VAE score:" + str(result_dict))
        metrics["FactorVAE"] = result_dict
    if MIG:
        from attribute_disentanglement.metrics.mig import compute_mig
        from attribute_disentanglement.metrics.utils import _histogram_discretize
        result_dict = compute_mig(
            dataset,
            _representation,
            random_state=np.random.RandomState(0),
            artifact_dir=None,
            num_train=10000, 
            batch_size=64,
            discretizer_num_bins=20,
            discretizer_fn=_histogram_discretize,
        )
        print("MIG score:" + str(result_dict))
        metrics["MIG"] = result_dict

    return metrics


def evaluate_segmentation(encoder, val_loader):
    from attribute_disentanglement.metrics.ocl.test_seg import eval_masks, AverageMeter
    from tqdm import tqdm
    results = {}
    for data_dict in tqdm(val_loader):
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        _, pred_masks = encoder.encode(data_dict['img'])  # [B, N(, 1), H, W]
        if len(pred_masks.shape) == 5:
            pred_masks = pred_masks.squeeze(-3)
        gt_masks = data_dict['masks'].long()  # [B, H, W]
        # on VOC and COCO, some pixels are ignored during evaluation
        inst_overlap_masks = data_dict.get('inst_overlap_masks', None)
        mask_dict = eval_masks(pred_masks, gt_masks, inst_overlap_masks)
        # on VOC and COCO, we compute metrics with semantic and instance masks
        # if val_loader.dataset.lower() in ['coco', 'voc']:
        if 'sem_masks' in data_dict:
            mask_dict = {f'inst/{k}': v for k, v in mask_dict.items()}
            gt_sem_masks = data_dict['sem_masks']
            sem_mask_dict = eval_masks(
                pred_masks, gt_sem_masks, inst_overlap_masks,
            )
            sem_mask_dict = {f'sem/{k}': v for k, v in sem_mask_dict.items()}
            mask_dict.update(sem_mask_dict)

        B = pred_masks.shape[0]
        if not results:
            results = {k: AverageMeter() for k in mask_dict}
        for k, v in mask_dict.items():
            results[k].update(v, B)

        torch.cuda.empty_cache()

    for k, v in results.items():
        print(f'{k}: {(v.avg * 100.):.2f}')

    return {k: v.avg for k, v in results.items()}
