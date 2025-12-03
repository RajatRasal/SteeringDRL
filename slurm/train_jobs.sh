#!/bin/bash

CONFIG_DIR="configs_logsnr/shapes3d"
SEEDS=4
BATCH_SIZE=128
EXP_NAME=vdmpp/shapes3d_low_discrep

for config in "$CONFIG_DIR"/*; do
    base=$(basename "$config")
    filename="${base%.*}"
    for seed in $(seq 0 $SEEDS); do
        echo "$seed$filename"
        sbatch -J "$seed-$filename" \
            -p gpus48 \
            ./slurm/train_base.sh "$seed$filename" $BATCH_SIZE $config $seed $EXP_NAME
    done
done