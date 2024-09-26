# Imports
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from segment_anything import sam_model_registry
import cv2
import sys

# Command Line arguments
window = sys.argv[1]
threshold = sys.argv[2]

# Load SAM model and prepare for encoding
sam_checkpoint = "/fp/projects01/ec332/data/models/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
sam.eval()

# SAM is not designed for binary data, so define a function for preprocessing
def preprocess_binary_matrix(binary_matrix):
    binary_rgb = np.stack([binary_matrix] * 3, axis=-1)  # Shape becomes (height, width, 3)
    binary_rgb = binary_rgb.astype(np.uint8)
    resized_binary = cv2.resize(binary_rgb, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    input_tensor = torch.tensor(resized_binary).permute(2, 0, 1).float() / 255.0  # Shape [3, 1024, 1024]
    return input_tensor


# Modify SAM to extract latent spaces
class SAMExtractLatentSpace(torch.nn.Module):
    def __init__(self, sam):
        super(SAMExtractLatentSpace, self).__init__()
        self.sam = sam
    def forward(self, x):
        encoder_output = self.sam.image_encoder(x)
        latent_space = encoder_output
        return latent_space

sam_extract_latent = SAMExtractLatentSpace(sam)

# Custom Dataset to load binary spectrograms
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms):
        self.spectrograms = spectrograms

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        bin_spec = self.spectrograms[idx]
        input_tensor = preprocess_binary_matrix(bin_spec)
        return input_tensor

# Load binary data
filepath_in = f'spectrograms/bin_spec_{window}_{threshold}.npz'
bin_data = np.load(filepath_in)
bin_specs = bin_data['spectrograms']

# Create Dataset and DataLoader
dataset = SpectrogramDataset(bin_specs)
batch_size = 8  # You can adjust the batch size
num_workers = 16  # Number of workers for parallel data loading

dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Extract latent spaces
latent_spaces = []
for batch in dataloader:
    batch = batch.to(device)
    with torch.no_grad():
        latent_space = sam_extract_latent(batch)
        latent_spaces.append(latent_space.cpu().numpy())

# Convert the list of latent spaces to a numpy array
latent_spaces = np.concatenate(latent_spaces, axis=0)
    
# Save latent spaces to .npz file
filepath_out = f'/fp/projects01/ec332/data/latent_spaces/file_{window}_{threshold}_sam_3.npz'
np.savez(filepath_out, latent_spaces=latent_spaces)
