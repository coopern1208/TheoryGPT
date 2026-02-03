#!/bin/bash

#SBATCH -J data_generation
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -t 10:00:00
#SBATCH --mem=32G
#SBATCH -o log/output_%j.txt
#SBATCH -e log/error_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=qian_niu@brown.edu

# Set the working directory to the project directory
cd /users/qniu3/physics/RL_model_builder_6.0

source venv/bin/activate
python grammar/gen_theory.py

# Capture the exit code
EXIT_CODE=$?

echo "========================================="
echo "Training run completed at $(date)"
echo "Exit code: $EXIT_CODE"

# Exit with the same code as the Python script
exit $EXIT_CODE