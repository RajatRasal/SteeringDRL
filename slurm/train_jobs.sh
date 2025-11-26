#!/bin/bash

CONFIG_DIR="configs/shapes3d/vp"
SEEDS=4
BATCH_SIZE=128
EXP_NAME=vp_arch_ema_stability_clipping

counter=0
for config in "$CONFIG_DIR"/config_*; do
    for seed in $(seq 0 $SEEDS); do
        echo $config $seed $counter
        sbatch -J "vp-v$counter-seed$seed" \
            -p gpus24,gpus48 \
            ./slurm/train_base.sh "vp-v$counter-seed$seed" $BATCH_SIZE $config $seed $EXP_NAME
    done
    ((counter++))
done