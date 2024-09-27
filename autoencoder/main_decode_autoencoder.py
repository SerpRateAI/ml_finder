import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from util_model import ConvAutoencoder, WeightedBinaryCrossEntropyLoss

# Main script to decode latent spaces and save decoded images
if __name__ == '__main__':
    # Parse command-line arguments
    window = int(sys.argv[1])
    threshold = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    weight = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    batch_size = int(sys.argv[6])

    # Check for bottle argument
    if len(sys.argv) > 7:
        bottle = int(sys.argv[7])
    else:
        bottle = 2  # Default value for bottle if not provided

    # Load the latent spaces from the .npz file
    latent_file = f"/fp/projects01/ec332/data/latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz"
    latent_spaces = np.load(latent_file)['all_latent_spaces']

    # Convert latent spaces to torch tensor
    latent_tensor = torch.tensor(latent_spaces, dtype=torch.float32)

    # Instantiate the full ConvAutoencoder model (same class as for encoding)
    model = ConvAutoencoder(bottle=bottle)

    # Load the trained model
    model_path = f'/fp/projects01/ec332/data/models/file_no_res_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Only use the decoder part of the model
    model.eval()

    # Prepare the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Decode latent spaces to get the reconstructed images using the decoder part of the model
    decoded_images = []
    with torch.no_grad():
        for latent in DataLoader(TensorDataset(latent_tensor), batch_size=1):
            latent_batch, = latent
            latent_batch = latent_batch.to(device)
            latent_batch = latent_batch.squeeze(1)
            decoded_image = model.decoder(latent_batch).cpu().numpy()
            decoded_images.append(decoded_image)

    # Convert to numpy array for saving
    decoded_images = np.array(decoded_images)
    print(f"Shape of decoded images: {decoded_images.shape}")

    # Define output file to save the decoded images to .npz
    output_file = f'/fp/projects01/ec332/data/decoded_images/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz'
    np.savez(output_file, decoded_images=decoded_images)

    print(f"Decoded images saved to .npz file: {output_file}")
