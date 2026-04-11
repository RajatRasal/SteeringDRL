#!/bin/bash
#SBATCH --output=./slurm_logs/slurm.%N.%j.log
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

export PYTHONPATH=/anvme/workspace/b143dc45-drl/Diffusion-Representation-Learning/experiments/SlotDiffusion:$PYTHONPATH
export PATH=$PATH:/anvme/workspace/b143dc45-drl/conda_envs/slotdiffusion/bin

export HTTP_PROXY="http://proxy.rrze.uni-erlangen.de:80"
export HTTPS_PROXY="http://proxy.rrze.uni-erlangen.de:80"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# python slotdiffusion/img_based/test_comp_gen.py \
#     --params "slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128-dup1.py" \
#     --weight "checkpoint/sa_ldm_clevrtex_params-res128-dup1/models/epoch/model_200.pth" \
#     --bs 64
# echo FID
# python3 -m pytorch_fid "checkpoint/sa_ldm_clevrtex_params-res128-dup1/models/epoch/eval/gt_imgs" "checkpoint/sa_ldm_clevrtex_params-res128-dup1/models/epoch/eval/comp_imgs"


# echo =================
# echo SLOT DIFFUSION
# for i in $(seq 1 5); do
#     echo REPEAT $i
#     python slotdiffusion/img_based/test_seg.py \
#         --params "slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/sa_ldm_clevrtex_params-res128_r32-dup$i/models/epoch/model_200.pth" \
#         --bs 64
#     python slotdiffusion/img_based/test_recon.py \
#         --params "slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/sa_ldm_clevrtex_params-res128_r32-dup$i/models/epoch/model_200.pth" \
#         --bs 64
#     python slotdiffusion/img_based/test_comp_gen.py \
#         --params "slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/sa_ldm_clevrtex_params-res128_r32-dup$i/models/epoch/model_200.pth" \
#         --bs 64
#     echo FID
#     python3 -m pytorch_fid "checkpoint/sa_ldm_clevrtex_params-res128_r32-dup$i/models/epoch/eval/gt_imgs" "checkpoint/sa_ldm_clevrtex_params-res128_r32-dup$i/models/epoch/eval/comp_imgs"
#     echo DONE REPEAT $i
# done
# echo =================

# echo =================
# echo SLATE
# for i in $(seq 5 5); do
#     echo REPEAT $i
#     python slotdiffusion/img_based/test_seg.py \
#         --params "slotdiffusion/img_based/configs/slate/slate_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/slate_clevrtex_params-res128_r32-dup$i/models/epoch/model_200.pth" \
#         --bs 64
#     python slotdiffusion/img_based/test_recon.py \
#         --params "slotdiffusion/img_based/configs/slate/slate_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/slate_clevrtex_params-res128_r32-dup$i/models/epoch/model_200.pth" \
#         --bs 64
#     python slotdiffusion/img_based/test_comp_gen.py \
#         --params "slotdiffusion/img_based/configs/slate/slate_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/slate_clevrtex_params-res128_r32-dup$i/models/epoch/model_200.pth" \
#         --bs 64
#     echo FID
#     python3 -m pytorch_fid "checkpoint/slate_clevrtex_params-res128_r32-dup$i/models/epoch/eval/gt_imgs" "checkpoint/slate_clevrtex_params-res128_r32-dup$i/models/epoch/eval/comp_imgs"
#     echo DONE REPEAT $i
# done
# echo =================

# echo =================
# echo SLOT ATTENTION
# for i in $(seq 1 5); do
#     echo REPEAT $i
#     python slotdiffusion/img_based/test_seg.py \
#         --params "slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/sa_clevrtex_params-res128_r32-dup$i/models/model_125000.pth" \
#         --bs 64
#     python slotdiffusion/img_based/test_recon.py \
#         --params "slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/sa_clevrtex_params-res128_r32-dup$i/models/model_125000.pth" \
#         --bs 64
#     python slotdiffusion/img_based/test_comp_gen.py \
#         --params "slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128_r32-dup$i.py" \
#         --weight "checkpoint/sa_clevrtex_params-res128_r32-dup$i/models/model_125000.pth" \
#         --bs 64
#     echo FID
#     python3 -m pytorch_fid "checkpoint/sa_clevrtex_params-res128_r32-dup$i/models/eval/gt_imgs" "checkpoint/sa_clevrtex_params-res128_r32-dup$i/models/eval/comp_imgs"
#     echo DONE REPEAT $i
# done
# echo =================
# 

for i in $(seq 1 5); do
    python slotdiffusion/img_based/test_seg.py \
        --params "slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224-dup$i.py" \
        --weight "checkpoint/sa_ldm_dino_voc_params-res224-dup$i/models/model_70209.pth" \
        --bs 64
done