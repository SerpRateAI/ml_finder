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
import pandas as pd
import random
from collections import defaultdict

# Import from parent directory
sys.path.append(os.path.abspath('..'))
from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_binary_spectrogram 

print('Script is running')
# Command-line arguments
station = 'A01'
window = 72
threshold = 0
perplexity = int(sys.argv[1])
clusters = int(sys.argv[2])
bins = int(sys.argv[3])
init = 'random'

# Load condensed images
#condensed_images_path = f"/fp/projects01/ec332/data/rules_based_latent/retrained_bins_{bins}_{window}_{threshold}.npz"
condensed_images_path = f"/fp/projects01/ec332/data/rules_based_latent/bins_{bins}_{window}_{threshold}.npz"
condensed_images_data = np.load(condensed_images_path, allow_pickle=True)
condensed_images = condensed_images_data['all_condensed_images']

# Flatten the condensed images for t-SNE and clustering
# Use np.reshape(-1) to flatten each image into a 1D array
condensed_images_reshaped = [np.ravel(img) for img in condensed_images]
condensed_images_reshaped = np.array(condensed_images_reshaped)

print('Condensed Images Loaded')

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

# Perform t-SNE on the condensed images for visualization
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init=init)
condensed_tsne = tsne.fit_transform(condensed_images_reshaped)

# Perform K-means clustering on the condensed images
n_clusters = clusters  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(condensed_images_reshaped) # was condensed_images_reshaped

print('Finished Clustering and t-SNE')

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
times = []  # To store times for cumulative plotting

print('Preparing Images for Visualization')

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

    # Store the times for cumulative plotting
    times.append(full_power_spec.times[0])

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
max_points = 2000
num_data_points = condensed_tsne.shape[0]
if num_data_points > max_points:
    selected_indices = random.sample(range(num_data_points), max_points)
else:
    selected_indices = list(range(num_data_points))
condensed_tsne = condensed_tsne[selected_indices]
bin_spec_images = [bin_spec_images[i] for i in selected_indices]
full_power_images = [full_power_images[i] for i in selected_indices]
colors = [colors[i] for i in selected_indices]

# Prepare data for Bokeh
source = ColumnDataSource(data=dict(
    x=condensed_tsne[:, 0],
    y=condensed_tsne[:, 1],
    bin_spec_images=bin_spec_images,
    full_power_images=full_power_images,
    indices=[str(i) for i in range(len(bin_spec_images))],
    colors=colors  # Add colors for clusters
))

# Create the Bokeh plot
p = figure(title="Interactive t-SNE with Original and Full Power Spectrograms",
           tools="pan,wheel_zoom,reset,save",
           width=800, height=800,
           x_axis_label="t-SNE Dimension 1", y_axis_label="t-SNE Dimension 2")

# Add scatter plot with color coding based on clusters
p.scatter('x', 'y', size=10, source=source, fill_color='colors', line_color=None)

# Add hover tool with images (binary and full power)
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
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">Index: @indices</span>
        </div>
    </div>
    """)
p.add_tools(hover)

# Save the plot as an HTML file
output_file(f"/fp/projects01/ec332/data/tsne_plots/manual_latent_{window}_{threshold}_cluster_{perplexity}_{clusters}_{bins}_{window}_{threshold}.html")
save(p)

print('Bokeh Plot Saved')

# Group times by cluster labels
cluster_times = defaultdict(list)
for i, time in enumerate(times):
    cluster_times[cluster_labels[i]].append(time)

# Determine the number of clusters
n_clusters = len(cluster_times)

# Sort the cluster_times by cluster ID (i.e., cluster label)
sorted_cluster_times = dict(sorted(cluster_times.items()))

# Create a figure with subplots: 2 columns, n_clusters rows
fig, axes = plt.subplots(n_clusters, 3, figsize=(18, n_clusters * 4), gridspec_kw={'width_ratios': [1, 1, 1]})

# Find the global min and max time across all clusters
all_times = [pd.Timestamp(t) for sublist in sorted_cluster_times.values() for t in sublist]
global_min_time = min(all_times)
global_max_time = max(all_times)

# Find the global maximum cumulative count across all clusters
global_max_cumulative_count = max([len(sorted_cluster_times[cluster_id]) for cluster_id in sorted_cluster_times])

# Loop over each cluster to create subplots in sorted order
cluster_index = 0
for cluster_id, ax_row in zip(sorted_cluster_times.keys(), axes):
    # Cumulative Plot (Left Column)
    # Convert times to a pandas Series for cumulative counting
    time_series = pd.Series([pd.Timestamp(t) for t in sorted_cluster_times[cluster_id]])
    time_series = time_series.sort_values()
    cumulative_counts = np.arange(1, len(time_series) + 1)
    
    # Plot cumulative occurrences for the current cluster
    ax_cumulative = ax_row[0]
    ax_cumulative.plot(time_series, cumulative_counts, label=f'Cluster {cluster_index}', color=cluster_colors[cluster_id])
    
    # Customize each subplot
    ax_cumulative.set_title(f"Cluster {cluster_index} Cumulative Over Time")
    ax_cumulative.set_ylabel("Cumulative Count")
    ax_cumulative.grid(True)
    
    # Set the x-axis limits to the global min and max times
    ax_cumulative.set_xlim([global_min_time, global_max_time])
    
    # Set the y-axis limits to ensure all plots have the same y range
    ax_cumulative.set_ylim([0, global_max_cumulative_count])
    
    # Add legend
    ax_cumulative.legend()

    # Rotate x-ticks for the cumulative plot
    ax_cumulative.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_cumulative.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Major locator: every 2 days
    ax_cumulative.xaxis.set_minor_locator(mdates.HourLocator(interval=12))  # Minor locator: every 12 hours
    plt.setp(ax_cumulative.get_xticklabels(), rotation=45, ha='right')  # Rotate x-ticks for the cumulative plot
    
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

        # Extract the start and end time of the spectrogram
        start_time = pd.Timestamp(full_power_spec.times[0])
        end_time = pd.Timestamp(full_power_spec.times[-1])
        date_str = start_time.strftime('%Y-%m-%d')

        # Plot the full power spectrogram on the specified axis using its `.plot()` method
        full_power_spec.plot(ax=ax_spec, min_db=-20, max_db=30)
        
        # Set only the date above the spectrogram
        ax_spec.set_title(f"{date_str}")
        
        # Set the x-ticks and x-axis labels based on actual times
        num_times = len(full_power_spec.times)
        if num_times > 10:  # Only show a subset of x-ticks if there are many
            tick_interval = num_times // 10  # Show 10 x-ticks at most
            selected_times = full_power_spec.times[::tick_interval]
        else:
            selected_times = full_power_spec.times

        # Convert times to numerical values for x-ticks
        ax_spec.set_xticks([mdates.date2num(t) for t in selected_times])
        ax_spec.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))  # Format the x-axis ticks as hours, minutes, seconds

        # Rotate the x-ticks for better readability on this subplot
        plt.setp(ax_spec.get_xticklabels(), rotation=45, ha='right')
        
        # increment cluster index
    cluster_index += 1

# Adjust the layout to prevent overlap
plt.tight_layout()

# Show or save the plot
plt.savefig(f"/fp/projects01/ec332/data/cumulative_cluster_plots/manual_latent_{window}_{threshold}_cluster_{perplexity}_{clusters}_{bins}_{window}_{threshold}_cumulative_time.png")
print('Saved Cumulative Time Plot')

