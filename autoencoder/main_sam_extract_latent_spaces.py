# Imports
import torch
import numpy as np
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

# SAM does is not designed for binary data so define a function for preprocessing
def preprocess_binary_matrix(binary_matrix):
    binary_rgb = np.stack([binary_matrix] * 3, axis=-1)  # Shape becomes (height, width, 3)
    resized_binary = cv2.resize(binary_rgb, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    input_tensor = torch.tensor(resized_binary).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return input_tensor

# Modify Sam to extract latent spaces
class SAMExtractLatentSpace(torch.nn.Module):
    def __init__(self, sam):
        super(SAMExtractLatentSpace, self).__init__()
        self.sam = sam
    def forward(self, x):
        encoder_output = self.sam.image_encoder(x)
        latent_space = encoder_output
        return latent_space
sam_extract_latent = SAMExtractLatentSpace(sam)

# Load binary data
filepath_in = f'spectrograms/bin_specs_{window}_{threshold}.npz'  
bin_data = np.load(filepath_in)
bin_specs = bin_data['spectrograms']

# Extract latent spaces
latent_spaces = []
for bin_spec in bin_specs:
    input_tensor = preprocess_binary_matrix(bin_spec).to(device)
    with torch.no_grad():
        latent_space = sam_extract_latent(input_tensor)
        latent_spaces.append(latent_space.cpu().numpy())

# Convert the list of latent spaces to a numpy array
latent_spaces = np.array(latent_spaces)

# Save latent spaces to .npz file
filepath_out = f'encoded_latent_spaces/file_{window}_{threshold}_sam.npz'  
np.savez(filepath_out, latent_spaces=latent_spaces)
