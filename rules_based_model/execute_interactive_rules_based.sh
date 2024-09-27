#!/bin/bash
#SBATCH --job-name=rules_interactive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=ec332
#SBATCH --time=1:00:00
#SBATCH --output=autoencoder_script/%j.log

# Arguments
perplexity=$1
clusters=$2
bins=$3


echo "$perplexity"
echo "$clusters"
echo "$bins"

# Activate environment
source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

# Run the Python script
python main_interactive_rules_based.py "$perplexity" "$clusters" "$bins"
