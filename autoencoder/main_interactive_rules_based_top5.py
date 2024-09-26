'''
This script performs clustering on spectrogram data using t-SNE and K-means clustering, and visualizes the results.
It loads spectrogram data, applies dimensionality reduction (t-SNE), clusters the data (K-means), and visualizes
the top 5 clusters (with the smallest intra-cluster distances) using cumulative plots and Bokeh plots.
'''

import os
import sys
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20, Category10, Turbo256
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import matplotlib.dates as mdates
import pandas as pd
import random
from collections import defaultdict

sys.path.append(os.path.abspath('..'))
from utils_basic import SPECTROGRAM_DIR as indir
from utils_spec import get_spectrogram_file_suffix, get_spec_peak_file_suffix, read_binary_spectrogram

print('Script is running')

# ========== Command-line Arguments ==========
station = 'A01'
window = 72
threshold = 345
perplexity = int(sys.argv[1])
clusters = int(sys.argv[2])
bins = int(sys.argv[3])
init = 'random'
n_clusters = clusters

# ========== Load Data ==========
condensed_images_path = f'/fp/projects01/ec332/data/rules_based_latent/bins_{bins}_{window}_{threshold}.npz'
condensed_images_data = np.load(condensed_images_path, allow_pickle=True)
condensed_images = condensed_images_data['all_condensed_images']

# Flatten the condensed images for t-SNE and clustering
condensed_images_reshaped = np.array([np.ravel(img) for img in condensed_images])

print('Condensed Images Loaded')

binary_spectrograms_path = f'/fp/projects01/ec332/data/altered_spectrograms/bin_spec_red_res_{window}_{threshold}.npz'
binary_spectrogram_data = np.load(binary_spectrograms_path, allow_pickle=True)
bin_specs = binary_spectrogram_data['spectrograms']

full_power_spectrograms_path = f'/fp/projects01/ec332/data/altered_spectrograms/power_spec_{station}_{window}_{threshold}.npz'
full_power_data = np.load(full_power_spectrograms_path, allow_pickle=True)
full_power_spectrograms = full_power_data['spectrograms']

print('Full Power Spectrograms Loaded')

# ========== Dimensionality Reduction and Clustering ==========
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init=init, random_state=42)
condensed_tsne = tsne.fit_transform(condensed_images_reshaped)

kmeans = KMeans(n_clusters=clusters, random_state=42)
cluster_labels = kmeans.fit_predict(condensed_tsne) #using the original latent spaces performs similarly

print('Finished Clustering and t-SNE')

# ========== Define Clustering and Plotting Functions ==========

def get_cluster_colors(n_clusters):
    '''
    Generate a list of colors for clusters. Use different color banks depending on the mumber of clusters.
    For over 20 classes, the Turbo256 colorbank must be sampled from evenly; otherwise all of the classes
    appear blue.
    '''
    if n_clusters <= 10:
        palette = Category10[10]
    elif n_clusters <= 20:
        palette = Category20[20]
    else:
        step = len(Turbo256) // n_clusters
        palette = [Turbo256[i * step] for i in range(n_clusters)]
    return palette

def create_rgb_image_with_red(binary_matrix):
    '''
    Convert the binary spectrogram with red resonance lines to an RGB image.
    '''
    height, width = binary_matrix.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[binary_matrix == False] = [0, 0, 0]  # Black pixels
    rgb_image[binary_matrix == True] = [255, 255, 255]  # White pixels
    rgb_image[binary_matrix == 2] = [255, 0, 0]  # Red pixels for resonance lines
    return Image.fromarray(rgb_image)

# ========== Prepare Images for Bokeh Visualization ==========
bin_spec_images = []
full_power_images = []
times = []  # To store times for cumulative plotting. This is done during the bokeh vis step because it is convenient.

for binary_spec, full_power_spec in zip(bin_specs, full_power_spectrograms):
    img = create_rgb_image_with_red(binary_spec)
    img = ImageOps.flip(img)
    buffered = BytesIO()
    img.save(buffered, format='PNG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    bin_spec_images.append(f'data:image/png;base64,{img_base64}')

    # Plot full power spectrogram and convert it to base64 PNG
    fig, ax = plt.subplots()
    ax.set_title(str(full_power_spec.times[0])[:10])
    full_power_spec.plot(ax=ax, min_db=-10, max_db=40)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    full_power_images.append(f'data:image/png;base64,{img_base64}')
    
    # Store times for cumulative plotting
    times.append(full_power_spec.times[0])

# ========== Compute Cluster Metrics and Select Top 5 Clusters (Manual Selection is preferable at the moment)==========
# Compute intra-cluster distances for each cluster
intra_cluster_distances = []
for cluster_id in range(n_clusters):
    cluster_points = condensed_tsne[cluster_labels == cluster_id]  # Use condensed_tsne, not condensed_images_reshaped as 
    centroid = kmeans.cluster_centers_[cluster_id]
    avg_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
    intra_cluster_distances.append((cluster_id, avg_distance))

# Sort clusters by intra-cluster distances and select the top 5
intra_cluster_distances.sort(key=lambda x: x[1])
top_5_clusters = [cluster_id for cluster_id, _ in intra_cluster_distances[:5]]
print(top_5_clusters)

# manually select cluster ids
top_5_clusters = [14,16,33,0,23]

# ========== Generate Bokeh Plot ==========
palette_colors = get_cluster_colors(clusters)

# Assign colors to each cluster
colors = [palette_colors[label] for label in cluster_labels]

# Group times by cluster labels
cluster_times = defaultdict(list)
for i, time in enumerate(times):
    cluster_times[cluster_labels[i]].append(time)

# Prepare data for Bokeh
source = ColumnDataSource(data=dict(
    x=condensed_tsne[:, 0],
    y=condensed_tsne[:, 1],
    bin_spec_images=bin_spec_images,
    full_power_images=full_power_images,
    indices=[str(i) for i in range(len(bin_spec_images))],
    colors=colors,
    cluster_id=[str(cluster_labels[i]) for i in range(len(cluster_labels))] 
))

# Create the Bokeh plot
p = figure(title='Interactive t-SNE with Original and Full Power Spectrograms',
           tools='pan,wheel_zoom,reset,save',
           width=800, height=800,
           x_axis_label='t-SNE Dimension 1', y_axis_label='t-SNE Dimension 2')

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
            <span style="font-size: 17px; font-weight: bold;">Cluster ID: @cluster_id</span><br>
            <span style="font-size: 17px; font-weight: bold;">Index: @indices</span>
        </div>
    </div>
    """)
p.add_tools(hover)

# Save the Bokeh plot as an HTML file
output_file(f'/fp/projects01/ec332/data/tsne_plots/manual_latent_{window}_{threshold}_cluster_{perplexity}_{clusters}_{bins}_{window}_{threshold}.html')
save(p)

print('Bokeh Plot Saved')

# ========== Plotting Cumulative Time for Top 5 Clusters ==========
filtered_cluster_times = {cluster_id: cluster_times[cluster_id] for cluster_id in top_5_clusters}
n_clusters_to_plot = len(filtered_cluster_times)
fig, axes = plt.subplots(n_clusters_to_plot, 3, figsize=(18, n_clusters_to_plot * 4), gridspec_kw={'width_ratios': [1, 1, 1]})

all_times = [pd.Timestamp(t) for sublist in filtered_cluster_times.values() for t in sublist]
global_min_time, global_max_time = min(all_times), max(all_times)
global_max_cumulative_count = max(len(times) for times in filtered_cluster_times.values())

# Loop over the top 5 clusters and plot cumulative and spectrograms
for cluster_index, (cluster_id, ax_row) in enumerate(zip(filtered_cluster_times.keys(), axes)):
    # Cumulative Plot
    time_series = pd.Series([pd.Timestamp(t) for t in filtered_cluster_times[cluster_id]]).sort_values()
    cumulative_counts = np.arange(1, len(time_series) + 1)
    
    ax_cumulative = ax_row[0]
    ax_cumulative.plot(time_series, cumulative_counts, label=f'Cluster {cluster_id}', color=palette_colors[cluster_id])
    ax_cumulative.set_title(f'Cluster {cluster_id} Cumulative Over Time')
    ax_cumulative.set_ylabel('Cumulative Count')
    ax_cumulative.grid(True)
    ax_cumulative.set_xlim([global_min_time, global_max_time])
    ax_cumulative.set_ylim([0, global_max_cumulative_count])
    ax_cumulative.legend()
    ax_cumulative.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_cumulative.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_cumulative.xaxis.set_minor_locator(mdates.HourLocator(interval=12))
    plt.setp(ax_cumulative.get_xticklabels(), rotation=45, ha='right')

    # Spectrograms
    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    sample_indices = random.sample(cluster_indices, 2) if len(cluster_indices) >= 2 else cluster_indices * 2

    for i, ax_spec in enumerate(ax_row[1:3]):
        full_power_spec = full_power_spectrograms[sample_indices[i]]
        full_power_spec.plot(ax=ax_spec, min_db=-20, max_db=30)
        ax_spec.set_title(pd.Timestamp(full_power_spec.times[0]).strftime('%Y-%m-%d'))
        selected_times = full_power_spec.times[::max(1, len(full_power_spec.times) // 10)]
        ax_spec.set_xticks([mdates.date2num(t) for t in selected_times])
        ax_spec.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.setp(ax_spec.get_xticklabels(), rotation=45, ha='right')

# Adjust layout and save the cumulative plot
plt.tight_layout()
plt.savefig(f'/fp/projects01/ec332/data/cumulative_cluster_plots/top_5_manual_latent_{window}_{threshold}_cluster_{perplexity}_{clusters}_{bins}_{window}_{threshold}_cumulative_time.png')
print('Saved Cumulative Time Plot for Top 5 Clusters')

