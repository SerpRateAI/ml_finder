#!/bin/bash
#SBATCH --job-name=rules_interactive
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --partition=accel
#SBATCH --account=ec332
#SBATCH --time=5:00:00
#SBATCH --output=autoencoder_script/%j.log

# Arguments
bins=$1
epochs=$2

echo "$bins"

echo "Activating Source"
# Activate environment
source /fp/homes01/u01/ec-benm/SerpRateAI/MlFinder/bin/activate

echo "Source Activated"

# Run the Python script
python main_train_autoencoder_rules_based.py "$bins" "$epochs"
