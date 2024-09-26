import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from util_model import ConvAutoencoder, WeightedBinaryCrossEntropyLoss

# Main training script
# Parse command-line arguments
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

# Load binary spectrograms and convert to torch object
bin_specs_arr = np.load(f'/fp/projects01/ec332/data/altered_spectrograms/bin_spec_no_res_{window}_{threshold}.npz')['spectrograms']
data = torch.tensor(bin_specs_arr, dtype=torch.float32)
data = data.unsqueeze(1)
full_dataset = TensorDataset(data, data)  # Targets are the same as inputs

# Split the dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

# Prepare DataLoader for training
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

# Set device
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
plt.savefig('/fp/projects01/ec332/data/debugging/reconstruction_debug.png')
plt.close()