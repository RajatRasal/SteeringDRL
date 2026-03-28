#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --output=./slurm_logs_vqvae/slurm.%N.%j.log

source /vol/cuda/12.9.0/setup.sh
source .venv/bin/activate

# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# # 2) Give each job a unique DDP rendezvous (even if world_size=1)
# #    dist.init() often binds a store on MASTER_ADDR:MASTER_PORT.
# export MASTER_ADDR=127.0.0.1
# # Derive a stable-but-unique port from SLURM_JOB_ID (stay in 10000-19999)
# export MASTER_PORT=$((10000 + SLURM_JOB_ID % 5000))
# 
# # (Optional) If your code respects these, force single-process DDP:
# export WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0

# export PATH="/vol/biomedic3/rrr2417/encdiff/k-diffusion/.venv/bin:/vol/biomedic3/rrr2417/.local/bin:$PATH"

echo "=== Node: $(hostname) ==="
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

echo "--- PyTorch / CUDA info ---"
python3 - <<'PY'
import torch
print(f"torch version:        {torch.__version__}")
print(f"torch.version.cuda:   {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device count:    {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, "
              f"cap={torch.cuda.get_device_capability(i)}")
else:
    print("CUDA not available.")
PY

python3 train_vqvae.py \
    --out-dir=$1 \
    --dataset=$2 \
    --cbase=$3 \
    --duration=30 \
    --batch=$4 \
    --seed=$5 \
    --dump=10 \
    --lr=$6 \
    --disc=$7 \
    --res=$8 \
    --vocab=$9
