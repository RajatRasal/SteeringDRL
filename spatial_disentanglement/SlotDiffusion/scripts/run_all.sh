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

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a100 sa_ldm_clevrtex_params-res128_ctsigmoid_v5_nobias_bw_r34 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_ctsigmoid_v5_nobias_bw_r34.py \
#     --task img_based --fp16 --cudnn

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a100 sa_ldm_clevrtex_params-res128_ctsigmoid_v5_nobias_bw_r34_epspred \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_ctsigmoid_v5_nobias_bw_r34_epspred.py \
#     --task img_based --fp16 --cudnn

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a100 sa_ldm_clevrtex_params-res128_ctsigmoid_v5_nobias_bw2_r34 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_clevrtex_params-res128_ctsigmoid_v5_nobias_bw2_r34.py \
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

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 sa_ldm_dino_voc_params-res224_ctsigmoid_nobias \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224_ctsigmoid_nobias.py \
#     --task img_based --fp16 --cudnn

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 sa_ldm_dino_voc_params-res224_ctsigmoid_v5_nobias \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224_ctsigmoid_v5_nobias.py \
#     --task img_based --fp16 --cudnn
# 
# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a40 sa_ldm_dino_voc_params-res224_ctsigmoid_v5_nobias_bw \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224_ctsigmoid_v5_nobias_bw.py \
#     --task img_based --fp16 --cudnn

# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a100 sa_ldm_dino_voc_params-res224_ctsigmoid_v5_nobias_bw_vpred \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_voc_params-res224_ctsigmoid_v5_nobias_bw_vpred.py \
#     --task img_based --fp16 --cudnn

# COCO
# TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
#     ./scripts/dup_run_sbatch.sh a100 sa_ldm_dino_coco_params-res224 \
#     scripts/train.py none \
#     slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224.py \
#     --task img_based --fp16 --cudnn

TIME=24:00:00 GPUS=1 CPUS_PER_GPU=16 QOS=normal REPEAT=5 \
    ./scripts/dup_run_sbatch.sh a100 sa_ldm_dino_coco_params-res224_ctsigmoid_v5_bw \
    scripts/train.py none \
    slotdiffusion/img_based/configs/sa_ldm/sa_ldm_dino_coco_params-res224_ctsigmoid_v5_bw.py \
    --task img_based --fp16 --cudnn