#!/bin/bash
#SBATCH --job-name=gpu_make_tsne_plot
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --output=autoencoder_script/%j.log
#SBATCH --gpus=a100:1
#SBATCH --mem=128GB
#SBATCH --partition=accel
#SBATCH --account=ec332

window=$1
threshold=$2
epochs=$3
weight=$4
rate=$5
batch=$6
perplexity=$7
station=$8

echo $window
echo $threshold
echo $epochs
echo $weight
echo $rate
echo $batch

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_interactive_tsne.py "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch" "$perplexity" "$station"
