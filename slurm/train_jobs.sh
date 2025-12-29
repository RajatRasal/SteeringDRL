#!/bin/bash

CONFIG_DIR="configs_logsnr2/shapes3d"
SEEDS=3
BATCH_SIZE=128
EXP_NAME=vdmpp_deep_longer_stable2_improved/shapes3d

for config in "$CONFIG_DIR"/*.json; do
    echo $config
    base=$(basename "$config")
    filename="${base%.*}"
    for modelseed in $(seq 0 $SEEDS); do
        for trainseed in $(seq 0 $SEEDS); do
            sbatch -J "$modelseed-$trainseed-$filename" \
                -p gpus24,gpus48 \
                ./slurm/train_base.sh "$modelseed$trainseed$filename" $BATCH_SIZE $config $modelseed $EXP_NAME $trainseed
        done
    done
done