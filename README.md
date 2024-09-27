# Unsuperised Signal Classifcation

Geophone arrays in the Oman region yeilded a zoo of spectrogram signals. This repository contains methods for splitting these signals into classes.

## Rules Based Model Example

The rules based model is so far the most effective clustering models, to run make sure to clone this directory, have access to the Fox Supercluster, and follow the instructions below.

First cd into the process_spectrograms folder. Type out the command: 

sbatch execute_filtered_specs.sh A01 72 345             (station number, window size in seconds, activity threshold)

This will load spectrograms from the A01 as well as binary spectrograms into the Fox cluster data. Next enter:

sbatch execute_remove_resonances.sh 72 345              (window size in seconds, activity threshold)

This will load the removed resonance binary spectrograms to the Fox cluster. Training with stationary resonances removed prevents the model from picking up on the trivial resonance features. Note that there is currently not an option to train the rules based model without this step. 

After removing the resonances, cd .. into the parent directory then cd into the rules_based_model folder. You will then need to enter:

sbatch execute_rules_based_encode.sh 5             (number of bins)

This applies slope filters, sums by rows, and collects bins to manually create a lower dimensional space for clustering. Finally enter:

sbatch execute_interactive_rules_based_top5.sh 20 40 5 (perplexity, number of classes, number of bins)

This will output an interactive tsne plot to the folder: /fp/projects01/ec332/data/cumulative_cluster_plots
And a cumulative occurence plot of the selected top 5classes: /fp/projects01/ec332/data/tsne_plots/

Note that the tsne plot is a large file ~ 100 Mb and may take a while to load.

## Other Models

...
