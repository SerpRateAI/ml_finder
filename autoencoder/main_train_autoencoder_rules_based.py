import sys
from os.path import join
import os
from torchviz import make_dot

from numpy import bool_, linspace, zeros
import numpy as np

from pandas import Timestamp
from pandas import date_range

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from time import time
from multiprocessing import Pool
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
# from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import random

class ConvAutoencoder2(nn.Module):
    def __init__(self, bottle, inputs = 1):
        super(ConvAutoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output, latent

# Import Internal Modules 
sys.path.append(os.path.abspath('..'))

from utils_basic import SPECTROGRAM_DIR as indir
from utils_plot import plot_geo_total_psd_to_bin_array_spectrogram, save_figure

print("initializing file")

# Initialize file path constants
bins = int(sys.argv[1])
num_epochs = int(sys.argv[2])
window = 72
threshold = 345
learning_rate = .001

# Setting seed for reproducibility
seed = 42 
generator = torch.Generator().manual_seed(seed)

# Load binary spectrograms and convert to torch object
rules_latent_spaces = np.load(f'/fp/projects01/ec332/data/rules_based_latent/bins_{bins}_{window}_{threshold}.npz')['all_condensed_images']
data = torch.tensor(rules_latent_spaces, dtype=torch.float32)
data = data.unsqueeze(1)
full_dataset = TensorDataset(data, data)  # Targets are the same as inputs
print(data.shape)

print('data loaded')

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, and optimizer
model = ConvAutoencoder2(bottle=16).to(device)
criterion =  nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in full_dataset:
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
    if (epoch + 1) % 2 == 0:
        avg_loss = epoch_loss / len(full_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print("Training complete.")

# Calculate the latent spaces and store to list
all_latent_spaces = []
with torch.no_grad():
    for batch in DataLoader(full_dataset, batch_size=1): 
        inputs, _ = batch
        inputs = inputs.to(device)
        decoded, latent = model(inputs)
        latent_space = latent.cpu().numpy()
        all_latent_spaces.append(latent_space)

# Save the latent spaces to a .npz file
np.savez(f'/fp/projects01/ec332/data/rules_based_latent/retrained_bins_{bins}_{window}_{threshold}.npz', all_condensed_images=np.array(all_latent_spaces))
print("Latent spaces saved to .npz file.")
print('Saved Latent Spaces')

print(f'Shape of all_latent_spaces: {np.array(all_latent_spaces).shape}')

# Visualize a few reconstructions immediately after training
model.eval()
num_samples_to_plot = 5
fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 2 * num_samples_to_plot))
with torch.no_grad():
    for i in range(num_samples_to_plot):
        # Get a random sample from the training data
        sample, _ = full_dataset[i]
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
plt.savefig(f'/fp/projects01/ec332/data/debugging/reconstruction_debug_{num_epochs}_{learning_rate}_{window}_{threshold}.png')
plt.close()
        
print('Saved Figure')

# Print Model Architecture
input_size = (1, 200, 16)
dummy_input = torch.randn(1, *input_size).to(device)
output = model(dummy_input)
model_viz = make_dot(output, params=dict(model.named_parameters()))
model_viz.render("model_architecture", format="png")

torch.save(model, "model.pth")

print("rendered")

input_names = ['Sentence']
output_names = ['yhat']
torch.onnx.export(model, batch.text, 'rnn.onnx', input_names=input_names, output_names=output_names)

print('Onnx model')
