#!/bin/bash

CONFIG_DIR="configs_logsnr/shapes3d"
SEEDS=4
BATCH_SIZE=128
EXP_NAME=vdmpp/shapes3d_early_stage_sech_dynamic_rezero

for config in "$CONFIG_DIR"/*.json; do
    echo $config
    base=$(basename "$config")
    filename="${base%.*}"
    for trainseed in $(seq 0 $SEEDS); do
        sbatch -J "$trainseed-$filename" \
            -p gpus48 \
            ./slurm/train_base.sh "$trainseed$filename" $BATCH_SIZE $config 0 $EXP_NAME $trainseed
    done
done