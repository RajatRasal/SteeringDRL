#!/bin/bash

# CLEVRTEX
# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 sa_clevrtex_params-res128_r32 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa/sa_clevrtex_params-res128_r32.py \
#     --task img_based --fp16 --cudnn
# 
# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 slate_clevrtex_params-res128_r32 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/slate/slate_clevrtex_params-res128_r32.py \
#     --task img_based --fp16 --cudnn

TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
    ./scripts/dup_run_sbatch.sh a100 sa_ldm_clevrtex_params-res128_r34 \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_r34.py \
    --task img_based --fp16 --cudnn

TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
    ./scripts/dup_run_sbatch.sh a100 sa_ldm_clevrtex_params-res128_r34_ct_sigmoid_v5 \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_r34_ct_sigmoid_v5.py \
    --task img_based --fp16 --cudnn

TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
    ./scripts/dup_run_sbatch.sh a40 sa_ldm_clevrtex_params-res128 \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128.py \
    --task img_based --fp16 --cudnn

TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
    ./scripts/dup_run_sbatch.sh a40 sa_ldm_clevrtex_params-res128_ct_sigmoid_v5 \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_ct_sigmoid_v5.py \
    --task img_based --fp16 --cudnn