import os
import sys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20, Category10
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.dates as mdates
from collections import defaultdict
import pandas as pd
import random

# Import from parent directory
sys.path.append(os.path.abspath('..'))
from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_binary_spectrogram

# Command-line arguments
window = int(sys.argv[1])
threshold = int(sys.argv[2])
num_epochs = int(sys.argv[3])
weight = int(sys.argv[4])
learning_rate = float(sys.argv[5])
batch_size = int(sys.argv[6])
perplexity = int(sys.argv[7])
station = str(sys.argv[8])
bottle = int(sys.argv[9])
clusters = int(sys.argv[10])
init = 'random'

# Load latent spaces 
latent_spaces_path = f"/fp/projects01/ec332/data/latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz"
#latent_spaces_path = f"/fp/projects01/ec332/data/latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz"
latent_data = np.load(latent_spaces_path)
latent_spaces = latent_data['all_latent_spaces']

print('Latent')

# Load binary spectrograms with red resonance lines
binary_spectrograms_path = f"/fp/projects01/ec332/data/altered_spectrograms/bin_spec_red_res_{window}_{threshold}.npz"
binary_spectrogram_data = np.load(binary_spectrograms_path, allow_pickle=True)
bin_specs = binary_spectrogram_data['spectrograms']
print('Red Resonance')

# Load full power spectrograms
full_power_spectrograms_path = f'/fp/projects01/ec332/data/altered_spectrograms/power_spec_{station}_{window}_{threshold}.npz'
full_power_data = np.load(full_power_spectrograms_path, allow_pickle=True)
full_power_spectrograms = full_power_data['spectrograms']
print('Full Power')
print(full_power_spectrograms.shape)

# Load decoded images
decoded_images_path = f"/fp/projects01/ec332/data/decoded_images/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz"
decoded_images_data = np.load(decoded_images_path, allow_pickle=True)
decoded_images = decoded_images_data['decoded_images']

print('Finished Loading Decoded')
print(decoded_images.shape)

# Reshape latent spaces for clustering
latent_spaces_reshaped = latent_spaces.reshape(latent_spaces.shape[0], -1)

# Apply K-means clustering on the latent spaces
n_clusters = clusters  # You can adjust the number of clusters based on your data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(latent_spaces_reshaped)

# Perform t-SNE on the latent spaces for visualization
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init=init)
latent_tsne = tsne.fit_transform(latent_spaces_reshaped)

print('Finished Clustering')

# Function to convert the binary spectrogram with red resonance lines to an RGB image
def create_rgb_image_with_red(binary_matrix):
    height, width = binary_matrix.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Set black pixels (0 in binary_matrix)
    rgb_image[binary_matrix == False] = [0, 0, 0]   # Black

    # Set white pixels (1 in binary_matrix)
    rgb_image[binary_matrix == True] = [255, 255, 255]  # White

    # Set red pixels (2 in binary_matrix) for resonance lines
    rgb_image[binary_matrix == 2] = [255, 0, 0]  # Red

    return Image.fromarray(rgb_image)

# Prepare the images for Bokeh
bin_spec_images = []
full_power_images = []
decoded_img_base64_list = []  # Store base64 encoded decoded images
times = []  # To store times for cumulative plotting

# Store average pixels per Hz for each cluster
pixels_per_hz = defaultdict(list)

print(decoded_images.shape)
print(decoded_images[0])

for i, (binary_spec, full_power_spec) in enumerate(zip(bin_specs, full_power_spectrograms)):
    # Convert binary matrix with red resonance lines to an RGB image
    img = create_rgb_image_with_red(binary_spec)
    img = ImageOps.flip(img)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    bin_spec_images.append(f"data:image/png;base64,{img_base64}")

    # Plot the full_power_spec object and convert it to an image
    fig, ax = plt.subplots()
    ax.set_title(str(full_power_spec.times[0])[0:10])
    full_power_spec.plot(ax=ax, min_db=-10, max_db=40)

    # Convert the plot to a PNG image and encode it as base64
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    full_power_images.append(f"data:image/png;base64,{img_base64}")

    # Convert decoded image to a PNG base64-encoded string
    decoded_img_array = decoded_images[i][0, 0]
    
    decoded_img_array = (decoded_img_array * 255).astype(np.uint8)
    decoded_img = Image.fromarray(decoded_img_array, mode='L')
    decoded_img = ImageOps.flip(decoded_img)
    buf = BytesIO()
    decoded_img.save(buf, format="PNG")
    buf.seek(0)
    decoded_img_base64 = base64.b64encode(buf.getvalue()).decode()
    decoded_img_base64_list.append(decoded_img_base64)

    # Store the times for cumulative plotting
    times.append(full_power_spec.times[0])

    # Calculate the average number of True pixels per Hz
    num_rows = binary_spec.shape[0]  # Number of frequency bins (rows)
    avg_pixels_per_hz = np.sum(binary_spec) / num_rows
    pixels_per_hz[cluster_labels[i]].append(avg_pixels_per_hz)

# Create color palette and store colors for each cluster
cluster_colors = {}
if n_clusters <= 10:
    cluster_colors = {i: Category10[10][i] for i in range(n_clusters)}
elif n_clusters <= 20:
    cluster_colors = {i: Category20[20][i] for i in range(n_clusters)}

# Assign colors to each point based on cluster labels
colors = [cluster_colors[label] for label in cluster_labels]

print('Finished Encoding and Image Prep')

# Downsample
max_points = 1000
num_data_points = latent_tsne.shape[0]
if num_data_points > max_points:
    selected_indices = random.sample(range(num_data_points), max_points)
else:
    selected_indices = list(range(num_data_points))
latent_tsne = latent_tsne[selected_indices]
bin_spec_images = [bin_spec_images[i] for i in selected_indices]
full_power_images = [full_power_images[i] for i in selected_indices]
decoded_img_base64_list = [decoded_img_base64_list[i] for i in selected_indices]
colors = [colors[i] for i in selected_indices]

# Prepare data for Bokeh
source = ColumnDataSource(data=dict(
    x=latent_tsne[:, 0],
    y=latent_tsne[:, 1],
    bin_spec_images=bin_spec_images,
    full_power_images=full_power_images,
    decoded_images=[f"data:image/png;base64,{decoded_img_base64}" for decoded_img_base64 in decoded_img_base64_list],
    indices=[str(i) for i in range(len(bin_spec_images))],
    colors=colors  # Add colors for clusters
))

# Create the Bokeh plot
p = figure(title="Interactive t-SNE with Original, Full Power Spectrograms, and Decoded Images",
           tools="pan,wheel_zoom,reset,save",
           width=800, height=800,
           x_axis_label="t-SNE Dimension 1", y_axis_label="t-SNE Dimension 2")

# Add scatter plot with color coding based on clusters
p.scatter('x', 'y', size=10, source=source, fill_color='colors', line_color=None)

# Add hover tool with images (binary, full power, and decoded)
hover = HoverTool(tooltips="""
    <div>
        <div>
            <img
                src="@bin_spec_images" height="150" alt="@indices" width="150"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
            <img
                src="@full_power_images" height="150" alt="@indices" width="150"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
            <img
                src="@decoded_images" height="150" alt="@indices" width="150"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">Index: @indices</span>
        </div>
    </div>
    """)
p.add_tools(hover)

# Save the plot as an HTML file
output_file(f"/fp/projects01/ec332/data/tsne_plots/file_red_res_{station}_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}_cluster_{perplexity}_{clusters}_mod.html")
save(p)

print('Bokeh Plot Saved')

binary_spectrograms_path = f"/fp/projects01/ec332/data/altered_spectrograms/bin_spec_no_res_{window}_{threshold}.npz"
binary_spectrogram_data = np.load(binary_spectrograms_path, allow_pickle=True)                          # renaming of variables here (warning)
bin_specs = binary_spectrogram_data['spectrograms']

### Average Pixels per Hz Plot per Cluster (Subplots) ###
# Create a figure with subplots, one for each cluster
fig, axes = plt.subplots(n_clusters, 1, figsize=(12, n_clusters * 3), sharex=True)

# Loop over each cluster to create subplots
for cluster_id in range(n_clusters):
    # Get the indices of all images in this cluster
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    
    # Initialize an array to accumulate pixel counts per frequency row (Hz) across all images in this cluster
    frequency_pixel_counts = np.zeros(bin_specs[0].shape[0])  # Shape[0] gives the number of frequency rows
    
    # Total number of images in this cluster
    total_images_in_cluster = len(cluster_indices)
    
    # Loop over all images in this cluster
    for index in cluster_indices:
        # Get the binary spectrogram corresponding to the index
        binary_image = bin_specs[index]
        
        # Sum the True values (i.e., `1`s) per row (frequency row)
        frequency_pixel_counts += np.sum(binary_image, axis=1)
    
    # Calculate the average percentage of True pixels per row (Hz) across all images in this cluster
    average_percentage_per_hz = (frequency_pixel_counts / binary_image.shape[1]) / total_images_in_cluster * 100  # shape[1] gives the number of columns
    
    # Plot the result for this cluster in its own subplot
    ax = axes[cluster_id] if n_clusters > 1 else axes  # Handle the case where there is only 1 cluster
    ax.plot(range(len(average_percentage_per_hz)), average_percentage_per_hz, label=f'Cluster {cluster_id}', color=cluster_colors[cluster_id])
    ax.set_title(f'Cluster {cluster_id}')
    ax.set_ylabel("Avg % of True Pixels")
    ax.grid(True)
    
    # Only set x-axis label for the bottom plot
    if cluster_id == n_clusters - 1:
        ax.set_xlabel("Frequency (Hz)")

# Adjust the layout to prevent overlap
plt.tight_layout()

# Save or show the average pixels per Hz plot
plt.savefig(f"/fp/projects01/ec332/data/avg_pixels_per_hz_plots/file_{station}_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}_{clusters}_avg_pixels_per_hz_subplots.png")
plt.show()

print('Saved average pixel plot')

cluster_times = defaultdict(list)
# Group times by cluster labels
for i, time in enumerate(times):
    cluster_times[cluster_labels[i]].append(time)
# Determine the number of clusters
n_clusters = len(cluster_times)

# Create a figure with subplots: 2 columns, n_clusters rows
fig, axes = plt.subplots(n_clusters, 3, figsize=(18, n_clusters * 4), gridspec_kw={'width_ratios': [2, 1, 1]})

# Loop over each cluster to create subplots
for cluster_id, ax_row in zip(cluster_times.keys(), axes):
    # Cumulative Plot (Left Column)
    # Convert times to a pandas Series for cumulative counting
    time_series = pd.Series([pd.Timestamp(t) for t in cluster_times[cluster_id]])
    time_series = time_series.sort_values()
    cumulative_counts = np.arange(1, len(time_series) + 1)
    
    # Plot cumulative occurrences for the current cluster
    ax_cumulative = ax_row[0]
    ax_cumulative.plot(time_series, cumulative_counts, label=f'Cluster {cluster_id}', color=cluster_colors[cluster_id])
    
    # Customize each subplot
    ax_cumulative.set_title(f"Cluster {cluster_id} Cumulative Over Time")
    ax_cumulative.set_ylabel("Cumulative Count")
    ax_cumulative.grid(True)
    
    # Add legend
    ax_cumulative.legend()
    
    # Spectrograms (Right Columns)
    # Select two random indices of spectrograms from the current cluster
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    if len(cluster_indices) >= 2:
        sample_indices = random.sample(cluster_indices, 2)
    else:
        # If less than two samples, duplicate to avoid errors
        sample_indices = cluster_indices * 2
    
    # Display the two example spectrograms using the `.plot()` method
    for i, ax_spec in enumerate(ax_row[1:3]):  # ax_row[1] and ax_row[2]
        spectrogram_idx = sample_indices[i]
        full_power_spec = full_power_spectrograms[spectrogram_idx]
        
        # Plot the full power spectrogram on the specified axis using its `.plot()` method
        full_power_spec.plot(ax=ax_spec, min_db=-10, max_db=40)
        ax_spec.set_title(f"Example {i+1} from Cluster {cluster_id}")
        
        # Optionally, you can remove or customize additional labels if needed
        ax_spec.set_xlabel('')  # Clear default labels if necessary
        ax_spec.set_ylabel('')

# Set the x-axis label for the bottom left subplot
axes[-1, 0].set_xlabel("Time")

# Format the x-axis to show more intermediate times
axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[-1, 0].xaxis.set_major_locator(mdates.DayLocator(interval=5))  # Major locator: every 5 days
axes[-1, 0].xaxis.set_minor_locator(mdates.DayLocator(interval=1))  # Minor locator: every day

# Apply the same x-axis formatting to all cumulative subplots
for ax in axes[:, 0]:  # Apply to all first column subplots
    # Optional: Add grid lines for minor ticks
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

# Auto-format the date labels for better readability
plt.gcf().autofmt_xdate()

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show or save the plot
plt.savefig(f"/fp/projects01/ec332/data/cumulative_cluster_plots/file_{station}_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}_{clusters}_red_full_power_spectrograms.png")
print('Saved Cumulative Time Plot')



