#!/bin/bash
#SBATCH --job-name=rules_interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=ec332
#SBATCH --time=4:00:00
#SBATCH --output=autoencoder_script/%j.log

# Arguments
bins=$1

echo "$bins"

echo "Activating Source"
# Activate environment
source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

echo "Source Activated"

# Run the Python script
python main_rules_based_encode.py "$bins"
