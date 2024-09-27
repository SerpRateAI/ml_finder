#!/bin/bash
#SBATCH --job-name=2nd_tsne_sam
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=ec332
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder_script/%j.log

window=$1
threshold=$2
epochs=$3
weight=$4
rate=$5
batch=$6
perplexity=$7
station=$8
bottle=$9

echo $window
echo $threshold
echo $epochs
echo $weight
echo $rate
echo $batch

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_2nd_sam2_interactive_tsne.py "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch" "$perplexity" "$station" "$bottle"
