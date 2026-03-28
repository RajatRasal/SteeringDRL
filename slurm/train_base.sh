#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --output=./slurm_logs/slurm.%N.%j.log

source /vol/cuda/12.9.0/setup.sh

source .venv/bin/activate

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

python3 train2.py \
    --name=$1 \
    --batch-size=$2 \
    --config=$3 \
    --seed=$4 \
    --exp-name=$5 \
    --sample-n=16 \
    --demo-every=2500 \
    --evaluate-every=2500 \
    --debug-log-every=100 \
    --save-every=50000 \
    --end-nimgs=15000000 \
    --training-seed=$6 \
    --gradient-clipping \
    --sampling-steps=50 \
    --eta=1.0


#     --demo-every=1000 \
#     --evaluate-every=1000 \
# Gives better metrics for latent_units = 20, but worse visuals
# --gradient-clipping
# Prefer using latent_units = 10 without gradient clipping