#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --output=./slurm_logs/slurm.%N.%j.log
#SBATCH --partition=gpus24,gpus48

source /vol/cuda/12.9.0/setup.sh

export PATH="/vol/biomedic3/rrr2417/miniconda3/envs/encdiff2/bin:/vol/biomedic3/rrr2417/miniconda3/condabin:/vol/biomedic3/rrr2417/.local/bin:$PATH"

# python3 main_val.py --base configs/autoencoder/shapes3d_vq_4_16.yaml --seed 0 -t --gpus 0,
# python3 main_val.py --base configs/autoencoder/cars3d_vq_4_16.yaml --seed 0 -t --gpus 0,
# python3 main_val.py --base configs/autoencoder/mpi3d_vq_4_16.yaml --seed 0 -t --gpus 0,

# python3 main_val.py --base configs/latent-diffusion/shapes3d-vq-4-16-encdiff.yaml --seed $1 -t --gpus 0,
python3 main_val.py --base configs/latent-diffusion/cars3d-vq-4-16-encdiff.yaml --seed $1 -t --gpus 0,
# python3 main_val.py --base configs/latent-diffusion/mpi3d-vq-4-16-encdiff.yaml --seed $1 -t --gpus 0,