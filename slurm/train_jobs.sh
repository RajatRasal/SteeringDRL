#!/bin/bash

CONFIG_DIR="configs_paper/mpi3d"
SEEDS=9
BATCH_SIZE=128
EXP_NAME=paper/mpi3d

for config in "$CONFIG_DIR"/*.json; do
    base=$(basename "$config")
    filename="${base%.*}"
    for seed in $(seq 0 $SEEDS); do
        # for modelseed in $(seq 0 $SEEDS); do
        #     for trainseed in $(seq 0 $SEEDS); do
        # sbatch -J "$modelseed-$trainseed-$filename" \
        #     -p gpus24,gpus48 \
        #     ./slurm/train_base.sh "$modelseed$trainseed$filename" $BATCH_SIZE $config $modelseed $EXP_NAME $trainseed
        sbatch -J "$seed-$filename" \
            -p gpus24,gpus48 \
            ./slurm/train_base.sh "$seed$seed$filename" $BATCH_SIZE $config $seed $EXP_NAME $seed
    done
done