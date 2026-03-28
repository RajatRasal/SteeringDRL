#!/bin/bash

TRAINING_DIR="ckpts/vqvae"
MAX_SEEDS=0
CBASE=32
BATCH=128
LR=5e-5
DISC=True
RES=64
VOCAB=2048

NODES=(mira09)
NUM_NODES=${#NODES[@]}

job_count=0

# for dataset in shapes3d mpi3d cars3d; do
#     for seed in $(seq 0 $MAX_SEEDS); do
#         # Pick node in round-robin fashion
#         node=${NODES[$((job_count % NUM_NODES))]}
# 
#         echo "Submitting dataset=$dataset seed=$seed to $node"
# 
#         sbatch -J "vqvae-$dataset-$seed" \
#                -p gpus24 \
#                --nodelist=$node \
#                ./slurm/slurm_vqvae_base.sh $TRAINING_DIR $dataset $CBASE $BATCH $seed $LR $DISC $RES $VOCAB
# 
#         # Increment global job counter
#         ((job_count++))
#     done
# done

sbatch -J "vqvae-clevrtex-0" \
    -p gpus48 \
    ./slurm/train_vqvae_base.sh $TRAINING_DIR clevrtex 64 64 0 3e-4 True 128 4096