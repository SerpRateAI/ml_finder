import sys
from os.path import join
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

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

# Load binary spectrograms
bin_specs_arr = np.load(f'/fp/projects01/ec332/data/altered_spectrograms/bin_spec_no_res_{window}_{threshold}.npz')['spectrograms']
print(f'Loaded spectrogram shape: {bin_specs_arr.shape}')  # Debug: Print the shape of the loaded data

# Convert to torch tensor and add a channel dimension
# Since the images are binary and 2D, add the channel dimension to make them (num_samples, 1, height, width)
bin_specs_tensor = torch.tensor(bin_specs_arr, dtype=torch.float32).unsqueeze(1)  # Shape (num_samples, 1, height, width)
print(f'Data tensor shape after adding channel dimension: {bin_specs_tensor.shape}')  # Debug: Print the shape of data_tensor

# Define custom dataset class for rotations but keep the targets as original spectrograms
class RotatedTensorDataset(torch_data.Dataset):
    def __init__(self, tensor_data):
        self.tensor_data = tensor_data

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        original_image = self.tensor_data[idx]  # Shape: (1, H, W)

        # Ensure the original image has 3 dimensions
        assert original_image.dim() == 3, f"Expected original_image to have 3 dimensions (C, H, W), but got {original_image.dim()}."

        # Create rotated versions of the image
        rotated_90 = torch.rot90(original_image, 1, [1, 2])
        rotated_180 = torch.rot90(original_image, 2, [1, 2])
        rotated_270 = torch.rot90(original_image, 3, [1, 2])

        # Stack the rotations along the channel dimension
        # Resulting shape: (4, H, W)
        images_tensor = torch.cat([original_image, rotated_90, rotated_180, rotated_270], dim=0)

        # Return input with shape (4, H, W) and target with shape (1, H, W)
        return images_tensor, original_image.squeeze(0)  # Return original without the extra channel

# Create the full dataset with the new RotatedTensorDataset
full_dataset = RotatedTensorDataset(bin_specs_tensor)

# Prepare data for PyTorch use
train_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, criterion, and optimizer
model = ConvAutoencoder(bottle=bottle_filters, inputs = 4).to(device)
pos_weight = weight
neg_weight = 1.0
criterion = WeightedBinaryCrossEntropyLoss(pos_weight=pos_weight, neg_weight=neg_weight)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, target in train_dataloader:  # inputs shape: (batch_size, 4, H, W)
        inputs = inputs.to(device)  # Now shape (batch_size, 4, H, W)
        target = target.to(device)  # Shape (batch_size, H, W)

        # Add a batch dimension to the target to match the input shape
        targets = target.unsqueeze(1)  # Shape: (batch_size, 1, H, W)

        # Forward pass
        outputs, latent = model(inputs)

        # Ensure the criterion receives the correct input
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

# Extract latent and decoded images
all_latent_spaces = []
all_decoded_spaces = []

model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        inputs, _ = batch  # Only need inputs for inference
        inputs = inputs.to(device)  # Shape (batch_size, 4, H, W)

        # Pass through the model to get latent and reconstructed images
        reconstructed, latent = model(inputs)

        # Collect latent spaces and decoded images
        all_latent_spaces.append(latent.cpu().numpy())
        all_decoded_spaces.append(reconstructed.cpu().numpy())

# Concatenate all collected latent spaces and decoded images
all_latent_spaces = np.concatenate(all_latent_spaces, axis=0)
all_decoded_spaces = np.concatenate(all_decoded_spaces, axis=0)

# Save the latent spaces to a .npz file
np.savez(f"/fp/projects01/ec332/data/latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle_filters}.npz", all_latent_spaces=all_latent_spaces)
print("Latent spaces saved to .npz file.")

# Save the decoded spaces to a .npz file
np.savez(f'/fp/projects01/ec332/data/decoded_images/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle_filters}.npz', decoded_images=all_decoded_spaces)
print("Decoded spaces saved to .npz file.")

# Visualize a few reconstructions immediately after training
num_samples_to_plot = 5
fig, axes = plt.subplots(num_samples_to_plot, 5, figsize=(12, 2 * num_samples_to_plot))  # 5 columns: 1 original + 4 reconstructions

with torch.no_grad():
    for i in range(num_samples_to_plot):
        # Get a random sample from the full dataset
        sample, original = full_dataset[i]
        
        # Ensure the input to the model has the correct shape
        sample_flat = sample.to(device)  # Shape (4, H, W)

        # Pass through the model
        reconstructed, _ = model(sample_flat.unsqueeze(0))  # Add batch dimension

        # Move to CPU and detach from computation graph
        original = original.cpu().numpy()  # Shape: (H, W)
        reconstructed = reconstructed.cpu().squeeze(0)  # Shape: (4, H, W)

        # Plot the original and reconstructed images
        axes[i, 0].imshow(sample[0], cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(reconstructed[0], cmap='gray')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')

# Save the figure to the specified directory
plt.tight_layout()
plt.savefig(f'/fp/projects01/ec332/data/debugging/reconstruction_debug_{window}_{threshold}_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle_filters}.png')
plt.close()


# Save the figure to the specified directory
plt.tight_layout()
plt.savefig(f'/fp/projects01/ec332/data/debugging/reconstruction_debug_{window}_{threshold}_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle_filters}.png')
plt.close()

# Save Model
model_save_path = f'/fp/projects01/ec332/data/models/file_no_res_{window}_{threshold}_model_{num_epochs}_{pos_weight}_{learning_rate}_{batch_size}_{bottle_filters}.pth'
torch.save(model.state_dict(), model_save_path)

print('finished')

