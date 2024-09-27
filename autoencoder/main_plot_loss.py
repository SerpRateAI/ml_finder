import sys
from os.path import join
import os

import numpy as np
import matplotlib.pyplot as plt

# Command-line arguments
window = int(sys.argv[1])
threshold = int(sys.argv[2])
num_epochs = int(sys.argv[3])
pos_weight = int(sys.argv[4])
learning_rate = float(sys.argv[5])
batch_size = int(sys.argv[6])
bottle_filters = int(sys.argv[7])


# Load the training and validation loss data from the .npz file
loss_file_path = f"model_loss/file_no_res_gpu_{window}_{threshold}_model_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}_{bottle_filters}.npz"
loss_data = np.load(loss_file_path)

# Extract the training and validation loss arrays
train_loss = loss_data['train_loss']
val_loss = loss_data['val_loss']

# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss\nModel: {num_epochs} epochs, LR: {learning_rate}, Batch Size: {batch_size}, Bottle: {bottle_filters}')
plt.legend()
plt.grid(True)

# Save and display the plot
plot_save_path = f"model_loss/loss_plot_{window}_{threshold}_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}_{bottle_filters}.png"
plt.savefig(plot_save_path)
plt.show()
