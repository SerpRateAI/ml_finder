'''
This script processes binary spectrogram data using various convolutional filters, 
performs slope, acceleration, and deceleration filtering, and condenses the resulting images 
by summing them into frequency bins. The final condensed images are saved in an `.npz` file.
'''

import torch
import torch.nn as nn
import numpy as np
import math
import os
import sys

print('Initializing...')

# Load spectrogram data
window = 72
threshold = 0
bin_specs_arr = np.load(f'/fp/projects01/ec332/data/altered_spectrograms/bin_spec_no_res_{window}_{threshold}.npz')['spectrograms']
print(f'Loaded {len(bin_specs_arr)} Spectrograms')

# ========== Define Filter Parameters ==========
print('Defining filter parameters...')
slope_filter_size = 9
acceleration_filter_size = 5
deceleration_filter_size = 5
num_angles = 5
num_frequency_bins = int(sys.argv[1])
output_dir = '/fp/projects01/ec332/data/rules_based_latent'

# Calculate padding sizes for each filter
slope_padding = slope_filter_size // 2
acceleration_padding = acceleration_filter_size // 2
deceleration_padding = deceleration_filter_size // 2

# ========== Define Filter Creating Functions ==========

def create_slope_filter(angle, size=slope_filter_size):
    '''
    Create a slope filter of a given angle (in degrees) for a binary image.
    '''
    filter_tensor = np.zeros((size, size))

    if angle < 0:
        angle = 180 + angle
    
    for i in range(size):
        for j in range(size):
            x, y = i - size // 2, j - size // 2
            if x == 0 and y == 0:
                filter_tensor[i, j] = 100
            else:
                angle_radians = math.atan2(y, x)
                deg = math.degrees(angle_radians)
                filter_tensor[i, j] = (45 - np.abs(90 - min(np.abs(angle - deg), 
                                            np.abs(angle + 180 - deg), np.abs(angle - 180 - deg)))) / 45
    return torch.tensor(filter_tensor, dtype=torch.float32)

def create_acceleration_filter(size=acceleration_filter_size):
    '''
    Create an acceleration filter with negative values on the left 
    and positive values on the right, constant across height.
    '''
    gradient = np.linspace(-1, 1, size)
    filter_tensor = np.tile(gradient, (size, 1))
    return torch.tensor(filter_tensor, dtype=torch.float32)

def create_deceleration_filter(size=deceleration_filter_size):
    '''
    Create a deceleration filter with positive values on the left 
    and negative values on the right, constant across height.
    '''
    gradient = np.linspace(1, -1, size)
    filter_tensor = np.tile(gradient, (size, 1))
    return torch.tensor(filter_tensor, dtype=torch.float32)

def condense_image_by_frequency_bins(image, num_bins):
    '''
    Condense an image by summing rows into the specified number of frequency bins.
    '''
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    height = image.size(0)
    bin_size = height // num_bins
    condensed_image = [torch.sum(image[i * bin_size : (i + 1) * bin_size if i < num_bins - 1 else height, :], dim=0)
                       for i in range(num_bins)]
    
    condensed_image = torch.stack(condensed_image, dim=0)
    return torch.sum(condensed_image, dim=1)

# ========== Create and Stack Filters ==========
print('Creating slope filters...')
angles = np.linspace(-60, 60, num_angles)
print(f'Angles for slope filters: {angles}')

filters = [create_slope_filter(angle) for angle in angles]
conv_filters = torch.stack([f.unsqueeze(0) for f in filters], dim=0)

# # Define the connectivity filter, uncomment for larger filter
# connect_filter = torch.tensor([[[[1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1],
#                                  [1, 1, 10, 1, 1],
#                                  [1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1]]]], dtype=torch.float32)

connect_filter = torch.tensor([[[[1, 1, 1],
                                  [1, 10, 1],
                                  [1, 1, 1]]]], dtype=torch.float32)

# Define convolutional layers
print('Setting up convolutional layers...')
connect_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)
connect_layer.weight = nn.Parameter(connect_filter)

conv_layer = nn.Conv2d(in_channels=1, out_channels=len(angles), kernel_size=slope_filter_size, 
                       stride=1, padding=slope_padding, bias=False)
conv_layer.weight = nn.Parameter(conv_filters)

acceleration_filter = create_acceleration_filter()
deceleration_filter = create_deceleration_filter()

accel_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=acceleration_filter_size, 
                        stride=1, padding=acceleration_padding, bias=False)
decel_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=deceleration_filter_size, 
                        stride=1, padding=deceleration_padding, bias=False)

accel_layer.weight = nn.Parameter(acceleration_filter.unsqueeze(0).unsqueeze(0))
decel_layer.weight = nn.Parameter(deceleration_filter.unsqueeze(0).unsqueeze(0))

# ========== Perform Convolutions ==========
print('Beginning to process spectrograms...')
all_condensed_images = []

for idx, binary_image in enumerate(bin_specs_arr):
    if idx % 100 == 0:
        print(f'Processing spectrogram {idx}/{len(bin_specs_arr)}')

    input_tensor = torch.tensor(binary_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Apply connectivity filter
    connect_tensor = connect_layer(input_tensor)
    connect_tensor_mask = connect_layer(input_tensor)
    connect_tensor[connect_tensor_mask > 10] = 1
    connect_tensor[connect_tensor_mask <= 10] = 0

    # Apply slope filters
    output_tensor_mask = conv_layer(connect_tensor)
    output_tensor = conv_layer(connect_tensor) - 100
    output_tensor[output_tensor < 0] = 0

    # Print the sum of absolute values of each convolutional filter's output
    if idx == 5:
        for i in range(output_tensor.shape[1]):
            print(f'Filter {i} sum of absolute values: {torch.sum(torch.abs(output_tensor[0, i])):.4f}')

    # Weight by the tangent of the slope and sum
    combined_image = torch.zeros_like(output_tensor[0, 0])
    for i, angle in enumerate(angles):
        combined_image += np.tan(np.deg2rad(angle)) * output_tensor[0, i]
    
    combined_tensor = combined_image.unsqueeze(0).unsqueeze(0)

    # Apply acceleration and deceleration filters
    acceleration_image = accel_layer(combined_tensor).squeeze().detach().numpy()
    deceleration_image = decel_layer(combined_tensor).squeeze().detach().numpy()

    # Filter for white pixels
    acceleration_image[output_tensor_mask[0][0] < 50] = 0
    deceleration_image[output_tensor_mask[0][0] < 50] = 0

    # Apply ReLU-like transformation
    acceleration_image[acceleration_image < 0] = 0
    deceleration_image[deceleration_image < 0] = 0

    # Print the sum of absolute values of the acceleration image
    if idx ==5:
        print(f'Acceleration image sum of absolute values: {np.sum(np.abs(acceleration_image)):.4f}')
        print(f'Deceleration image sum of absolute values: {np.sum(np.abs(deceleration_image)):.4f}')

    # Collect and process images
    output_images_list = list(output_tensor[0])
    output_images_list.extend([acceleration_image, deceleration_image])
    output_images_list = [img.detach().numpy() if isinstance(img, torch.Tensor) else img for img in output_images_list]
    
    # Condense images by frequency bins
    condensed_images_list = [condense_image_by_frequency_bins(img, num_frequency_bins) for img in output_images_list]
    all_condensed_images.append(condensed_images_list)

print('Finished processing all spectrograms')

# ========== Save the Condensed Images ==========
print('Saving condensed images...')
output_path = os.path.join(output_dir, f'bins_{num_frequency_bins}_{window}_{threshold}.npz')
np.savez(output_path, all_condensed_images=all_condensed_images)
print(f'Condensed images saved to {output_path}')
print(f'Final shape of condensed images: {np.array(all_condensed_images).shape}')
