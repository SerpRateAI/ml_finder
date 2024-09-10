#!/bin/bash
#SBATCH --job-name=train_autoencoder
#SBATCH --time=10:00:00
#SBATCH --output=autoencoder_script/%j.log
#SBATCH --partition=accel
#SBATCH --gpus=a100:1   

window=$1
threshold=$2
epochs=$3
weight=$4
rate=$5
batch=$6

echo $window
echo $threshold
echo $epochs
echo $weight
echo $rate
echo $batch

source /fp/homes01/u01/ec-benm/SerpRateAI/MicroquakesEnv/bin/activate

python main_train_autoencoder.py "$window" "$threshold" "$epochs" "$weight" "$rate" "$batch"

