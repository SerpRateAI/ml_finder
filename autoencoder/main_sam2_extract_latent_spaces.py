import torch
import sys
import numpy as np
import cv2
sys.path.append('/fp/homes01/u01/ec-benm/SerpRateAI/ml_finder/externals/sam2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Command Line arguments
window = sys.argv[1]
threshold = sys.argv[2]

# Set Device configuration (use cpu if no gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SAM2 model (assuming you already installed SAM2 in your environment)
checkpoint = '/fp/homes01/u01/ec-benm/SerpRateAI/ml_finder/externals/sam2/checkpoints/sam2_hiera_tiny.pt'
model_cfg = "sam2_hiera_t.yaml"
sam_model = build_sam2(model_cfg, checkpoint)
sam_model = sam_model.to(device) 

# Create the SAM2 predictor
predictor = SAM2ImagePredictor(sam_model)

# Function to preprocess binary matrix into appropriate tensor format
def preprocess_binary_matrix(binary_matrix):
    # Ensure the binary matrix is of type uint8
    binary_rgb = np.stack([binary_matrix] * 3, axis=-1).astype(np.uint8)
    resized_binary = cv2.resize(binary_rgb, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    input_tensor = torch.tensor(resized_binary).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return input_tensor


# Function to extract latent spaces from SAM2 model
def extract_latent_space(input_tensor):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Pass through SAM2's image encoder to extract latent space
        latent_space = predictor.model.image_encoder(input_tensor)
    return latent_space

# Load your binary spectrogram data
filepath_in = f'spectrograms/bin_spec_no_res_{window}_{threshold}.npz' #modified for no res version
bin_data = np.load(filepath_in)
bin_specs = bin_data['spectrograms']

# # Extract latent spaces
# latent_spaces = []
# for bin_spec in bin_specs:
#     input_tensor = preprocess_binary_matrix(bin_spec).to(device)
#     with torch.no_grad():
#         latent_space = extract_latent_space(input_tensor)
#         vision_features = latent_space['spectrograms']  # Adjust key based on your model's output structure
#         latent_spaces.append(vision_features.cpu().numpy())  # Convert to NumPy array


# Extract latent spaces
latent_spaces = []
for bin_spec in bin_specs:
    input_tensor = preprocess_binary_matrix(bin_spec).to(device)
    with torch.no_grad():
        latent_space = extract_latent_space(input_tensor)
        first_key = next(iter(latent_space))
        tensor = latent_space[first_key]
        latent_spaces.append(tensor.cpu().numpy())  # Convert to NumPy array


# Save the latent spaces to a .npz file
filepath_out = f'/fp/projects01/ec332/data/latent_spaces/file_no_res_{window}_{threshold}_sam2.npz'
np.savez(filepath_out, latent_spaces=latent_spaces)

