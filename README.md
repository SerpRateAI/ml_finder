# Unsupervised Signal Classification

This repository contains methods for classifying spectrogram signals collected from geophone arrays in the Oman region. The geophone data has resulted in a wide variety of signal types, or a "zoo" of spectrograms, and we aim to categorize these signals using unsupervised clustering methods.

## Rules-Based Model Example

The rules-based model has proven to be the most effective clustering approach so far. To run the model, follow the steps below. Ensure that you have cloned this repository, have access to the Fox Supercluster, and meet the necessary environment prerequisites.

### Steps to Run the Model

1. **Navigate to the `process_spectrograms` folder:**

    ```bash
    cd process_spectrograms
    ```

2. **Load spectrograms:**
   
   Run the following command to load the spectrograms and binary spectrograms for the specified station (`A01`), using a given window size and activity threshold:

    ```bash
    sbatch execute_filtered_specs.sh A01 72 345
    ```
   - `A01`: Station number
   - `72`: Window size in seconds
   - `345`: Activity threshold

3. **Remove resonances:**

   To remove stationary resonance features, run the following command:

    ```bash
    sbatch execute_remove_resonances.sh 72 345
    ```

   This step removes resonance features and stores the modified binary spectrograms in the Fox cluster. **Note:** The current version of the rules-based model **requires** this step, and training without removing resonances is not supported.

4. **Navigate to the `rules_based_model` folder:**

    ```bash
    cd ../rules_based_model
    ```

5. **Apply slope filters and binning:**

   To create a lower-dimensional space for clustering, run the following command. It applies slope filters, sums rows, and bins the results manually:

    ```bash
    sbatch execute_rules_based_encode.sh 5
    ```
   - `5`: Number of bins

6. **Generate an interactive t-SNE plot:**

   Finally, run this command to generate an interactive t-SNE plot and a cumulative occurrence plot of the top 5 classes:

    ```bash
    sbatch execute_interactive_rules_based_top5.sh 20 40 5
    ```
   - `20`: Perplexity
   - `40`: Number of classes
   - `5`: Number of bins

   - The t-SNE plot will be saved in: `/fp/projects01/ec332/data/tsne_plots/`
   - The cumulative occurrence plot will be saved in: `/fp/projects01/ec332/data/cumulative_cluster_plots/`
