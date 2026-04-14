#!/bin/bash

for seed in $(seq 0 9)
do
    sbatch -J "$seed-cars3d-ed" ./vqvae_slurm.sh $seed
done