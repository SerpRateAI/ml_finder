#!/bin/bash
#SBATCH --job-name=plot_all_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
clusters=${10}

echo $window
echo $threshold
echo $epochs
echo $weight
echo $rate
echo $batch
echo $perplexity
echo $station

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_interactive_tsne.py "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch" "$perplexity" "$station" "$bottle" "$clusters"
