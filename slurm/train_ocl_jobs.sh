#!/bin/bash

CONFIG_DIR=configs_final/shapes3d
SEEDS=9
BATCH_SIZE=64
EXP_NAME=final/shapes3d

for config in "$CONFIG_DIR"/*.json; do
    echo $config
    base=$(basename "$config")
    filename="${base%.*}"
    for seed in $(seq 0 $SEEDS); do
        sbatch -J "$seed-$filename-shapes3d" \
            -p gpus24,gpus48 \
            ./slurm/train_ocl_base.sh "$seed$seed$filename" $BATCH_SIZE $config $seed $EXP_NAME $seed
    done
done