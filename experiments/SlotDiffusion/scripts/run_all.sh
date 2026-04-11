#!/bin/bash

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
# 
# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a100 sa_ldm_clevrtex_params-res128_r32 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_r32.py \
#     --task img_based --fp16 --cudnn

# # VOC
# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 sa_ldm_dino_voc_params-res224 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224.py \
#     --task img_based --fp16 --cudnn

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 sa_ldm_dino_voc_params-res224_ctsigmoid \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224_ctsigmoid.py \
#     --task img_based --fp16 --cudnn

TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
    ./scripts/dup_run_sbatch.sh a40 sa_ldm_dino_coco_params-res224_ctsigmoid \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224_ctsigmoid.py \
    --task img_based --fp16 --cudnn