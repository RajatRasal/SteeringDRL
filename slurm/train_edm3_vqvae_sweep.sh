#!/bin/bash
# Submit with:
#   sbatch slurm/train_edm3_vqvae_sweep.sh
#
# Array tasks 0-24 use the seed-2 VQ-VAE at 100,096 steps.
# Array tasks 25-49 use the seed-2 VQ-VAE at 150,000 steps.
# Within each block, tasks cover all 5 model seeds x 5 timestep seeds.

#SBATCH --job-name=edm3-vq-sweep
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus24,gpus48
#SBATCH --output=./slurm_logs/slurm.%N.%A_%a.log

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

source /vol/cuda/12.9.0/setup.sh
source .venv/bin/activate

task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
save_every=${SAVE_EVERY:-4500}
if (( task_id < 0 || task_id >= 50 )); then
    echo "Array task must be in [0, 49], got $task_id" >&2
    exit 2
fi

vq_index=$((task_id / 25))
seed_index=$((task_id % 25))
model_seed=$((seed_index / 5))
timestep_seed=$((seed_index % 5))

case "$vq_index" in
    0)
        vq_label="vq100k_s2"
        config="configs_logsnr/shapes3d/v_cosine_edm-3_vq100k_s2.json_"
        ;;
    1)
        vq_label="vq150k_s2"
        config="configs_logsnr/shapes3d/v_cosine_edm-3_vq150k_s2.json_"
        ;;
    *)
        echo "Unexpected VQ-VAE index: $vq_index" >&2
        exit 2
        ;;
esac

run_name="${model_seed}${timestep_seed}v_cosine_edm-3_${vq_label}"
exp_name="vdmpp/shapes3d_early_stage_edm3_${vq_label}"

echo "Node: $(hostname)"
echo "Array task: $task_id"
echo "VQ-VAE: $vq_label"
echo "Model seed: $model_seed"
echo "Timestep seed: $timestep_seed"
echo "Save every: $save_every steps ($((save_every * 128)) images)"
echo "Output: ckpts/$exp_name/$run_name"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

python3 -m attribute_disentanglement.scripts.train2 \
    --name="$run_name" \
    --batch-size=128 \
    --config="$config" \
    --seed="$model_seed" \
    --training-seed="$timestep_seed" \
    --exp-name="$exp_name" \
    --sample-n=16 \
    --demo-every=500 \
    --evaluate-every=500 \
    --debug-log-every=100 \
    --save-every="$save_every" \
    --end-nimgs=3000000 \
    --sampling-steps=50 \
    --eta=1.0
