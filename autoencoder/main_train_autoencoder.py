import sys
from os.path import join
import os

from numpy import bool_, linspace, zeros
import numpy as np

from pandas import Timestamp
from pandas import date_range

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from time import time
from multiprocessing import Pool
import h5py

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import random
from util_model import ConvAutoencoder, WeightedBinaryCrossEntropyLoss


# Import Internal Modules 
sys.path.append(os.path.abspath('..'))

from utils_basic import SPECTROGRAM_DIR as indir
from utils_plot import plot_geo_total_psd_to_bin_array_spectrogram, save_figure

print("initializing file")

# Initialize file path constants
window = int(sys.argv[1])
threshold = int(sys.argv[2])

# Set hyperparameters
if len(sys.argv) >= 7:
    num_epochs = int(sys.argv[3])
    weight = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    batch_size = int(sys.argv[6])
else:
    num_epochs = 800
    weight = 100
    learning_rate = .0015
    batch_size = 128

if len(sys.argv) >= 8:
    bottle_filters = int(sys.argv[7])
else: 
    bottle_filters = 2
print(f'bottle_filters: {bottle_filters}')

# Setting seed for reproducibility
seed = 42 
generator = torch.Generator().manual_seed(seed)

# Load binary spectrograms and convert to torch object
bin_specs_arr = np.load(f'/fp/projects01/ec332/data/altered_spectrograms/bin_spec_no_res_{window}_{threshold}.npz')['spectrograms']
# bin_specs_arr = np.load(f'spectrograms/bin_spec_{window}_{threshold}.npz')['spectrograms']
data = torch.tensor(bin_specs_arr, dtype=torch.float32)
data = data.unsqueeze(1)
full_dataset = TensorDataset(data, data)  # Targets are the same as inputs
print(full_dataset.shape)

print('data loaded')
# Get training data
train_size = .8
val_size = .2
train_data, val_data = random_split(full_dataset, [train_size, val_size]) # Only use train data

# Prepare data for pytorch use
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, and optimizer
model = ConvAutoencoder(bottle=bottle_filters).to(device)
pos_weight = weight
neg_weight = 1.0
criterion = WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight, neg_weight=neg_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs, latent = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("Training complete.")

# Visualize a few reconstructions immediately after training
model.eval()
num_samples_to_plot = 5
fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 2 * num_samples_to_plot))
with torch.no_grad():
    for i in range(num_samples_to_plot):
        # Get a random sample from the training data
        sample, _ = train_data[i]
        sample = sample.unsqueeze(0).to(device)  # Add batch dimension
        reconstructed, _ = model(sample)

        # Move to CPU and detach from computation graph
        sample = sample.cpu().squeeze().numpy()
        reconstructed = reconstructed.cpu().squeeze().numpy()

        # Plot the original and reconstructed images
        axes[i, 0].imshow(sample, cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed, cmap='gray')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')

# Save the figure to the specified directory
plt.tight_layout()
plt.savefig(f'/fp/projects01/ec332/data/debugging/reconstruction_debug_{window}_{threshold}_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle_filters}.png')
plt.close()

# Save Model
model_save_path = f'/fp/projects01/ec332/data/models/file_no_res_{window}_{threshold}_model_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}_{bottle_filters}.pth'
torch.save(model.state_dict(), model_save_path)
# # Find loss for validation data
#model.eval()
#for epoch in range(num_epochs):
#    epoch_loss = 0
#    for batch in val_dataloader:
#        inputs, targets = batch
#        inputs, targets = inputs.to(device), targets.to(device)
#        outputs, latent = model(inputs)
#        unbiased_loss = unbiased_criterion(outputs, targets)
#        epoch_loss += unbiased_loss.item()
#    avg_loss = epoch_loss / len(val_dataloader)
#    val_loss_list.append(avg_loss)
#print("Validation Complete")
#np.savez(f"model_loss/file_no_res_gpu_{window}_{threshold}_model_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}_{bottle_filters}.npz", train_loss=np.array(train_loss_list), val_loss= np.array(val_loss_list))
print('finished')
