#!/bin/bash
#SBATCH --job-name=sam_extract_latent
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder_script/%j.log
#SBATCH --gpus=a100:1
#SBATCH --mem=64GB
#SBATCH --partition=accel
#SBATCH --account=ec332

window=$1
threshold=$2

echo "Sam Extract Latent Spaces"
echo $window
echo $threshold

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_sam_extract_latent_spaces.py  "$window" "$threshold"
