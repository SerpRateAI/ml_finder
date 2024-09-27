#!/bin/bash
#SBATCH --job-name=sam2_extract_latent
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder_script/%j.log
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --partition=accel
#SBATCH --account=ec332

window=$1
threshold=$2
epochs=$3
weight=$4
rate=$5
batch=$6
bottle=$7

echo "Sam Extract Latent Spaces"
echo $window
echo $threshold
echo $epochs
echo $weight
echo $rate
echo $batch

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_2nd_sam2_extract_latent_spaces.py  "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch" "$bottle"
