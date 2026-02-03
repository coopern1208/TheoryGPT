#!/bin/bash

#SBATCH -J particle_graph_training
#SBATCH --account=lgouskos-gcondo
#SBATCH --partition=lgouskos-h100-gcondo
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH --mem=80G
#SBATCH -o log/output_%j.txt
#SBATCH -e log/error_%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qian_niu@brown.edu

cd /users/qniu3/physics/RL_model_builder_6.0

echo "Starting training run at $(date)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: H100 (1 GPU)"
echo "Memory: 80GB"
echo "Queue: lgouskos-h100-gcondo"
echo "Account: lgouskos-gcondo"
echo "Current directory: $(pwd)"
echo "Files in directory: $(ls -la)"
echo "========================================="

module load cuda/12.9.0-cinr
module load cudnn/9.8.0.87-12-y7fu

source venv/bin/activate

# Clear Python cache to ensure updated code is loaded
rm -rf __pycache__ architecture/__pycache__

python test_kv_cache.py

EXIT_CODE=$?
echo "========================================="
echo "GRPO training completed at $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
