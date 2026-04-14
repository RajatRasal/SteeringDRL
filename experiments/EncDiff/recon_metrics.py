import torch.nn.functional as F
import torch
from math import sqrt, ceil, exp
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np


def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(1).mean(1).mean(1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def calculate_ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel)

def calculate_lpips(img1, img2, lpips_fn):
    return lpips_fn.forward(img1, img2).view(-1)

def calculate_lpips(x, y, loss_fn):
    """x/y: [B, 3, H, W], torch.Tensor in [-1, 1]"""
    return loss_fn(x, y).mean()

def calculate_mse(img1, img2):
    return (img1 - img2).pow(2).mean(dim=[1, 2, 3])


from typing import List
import lpips
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from einops import rearrange, repeat
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# config = OmegaConf.load("configs/latent-diffusion/shapes3d-vq-4-16-enc-sin.yaml")
# config = OmegaConf.load("configs/latent-diffusion/shapes3d-vq-4-16-encdiff.yaml")
config = OmegaConf.load("configs/latent-diffusion/cars3d-vq-4-16-encdiff.yaml")
config.model.params.unet_config.params.latent_unit = 20
config.model.params.cond_stage_config.params.latent_unit = 20
model = instantiate_from_config(config.model)
# model.load_state_dict(torch.load("shapes3d_exp_aba/2023-12-20T15-15-04_shapes3d-vq-4-16-enc-sin363/checkpoints/last.ckpt")['state_dict'])
# model.load_state_dict(torch.load("logs2/2026-03-11T22-14-19_shapes3d-vq-4-16-encdiff0/checkpoints/last.ckpt")['state_dict'])
model.load_state_dict(torch.load("logs2/2026-03-11T22-19-52_cars3d-vq-4-16-encdiff0/checkpoints/last.ckpt")['state_dict'])
lpips_fn = lpips.LPIPS(net='vgg').cuda()
# config.data.params.batch_size = 1024
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()
i = 0

print("HERE")

model.cuda()
ddim_steps = 50
data_list = []
model.eval()
ssim_score_list = []
lpips_score_list = []
mse_score_list = []
for batch in tqdm(data._train_dataloader()):
    N = batch['image'].shape[0]
    x0 = batch['image'].cuda()
    # print(x0.min(), x0.max(), x0.mean())
    
    with torch.no_grad():
        z, c, x, xrec, xc, orc = model.get_input({'image':x0}, model.first_stage_key,
                                                    return_first_stage_outputs=True,
                                                    force_c_encode=True,
                                                    return_original_cond=True,
                                                    bs=N,return_false=True)
    
        samples, z_denoise_row = model.sample_log(cond=c,batch_size=N,ddim=True,
                                                    ddim_steps=ddim_steps,eta=1.)
        x_recon = model.decode_first_stage(samples)

    print(x_recon.min(), x_recon.max())

    # norm_x_0 = (x0 + 1.) / 2.
    # norm_x_recon = (x_recon + 1.) / 2.

    # ssim_score = calculate_ssim(norm_x_0, norm_x_recon.permute(0,2,3,1))
    lpips_score = calculate_lpips(x0.permute(0,3,1,2), x_recon, lpips_fn)
    # mse_score = calculate_mse(norm_x_0, norm_x_recon.permute(0,2,3,1))

    # ssim_score_list.append(ssim_score.mean().item())
    lpips_score_list.append(lpips_score.mean().item())
    # mse_score_list.append(mse_score.mean().item())
    if i % 20 == 0:
        # print(np.mean(ssim_score_list))
        # print(np.mean(lpips_score_list))
        print(np.mean(lpips_score_list))
        # print(np.mean(mse_score_list))
    i += 1