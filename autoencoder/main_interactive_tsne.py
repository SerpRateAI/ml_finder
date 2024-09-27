import sys
import os
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
#latent_spaces_path = f"encoded_latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz"
latent_spaces_path = f"/fp/projects01/ec332/data/latent_spaces/file_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}.npz"
latent_data = np.load(latent_spaces_path)
latent_spaces = latent_data['all_latent_spaces']

# Load binary spectrograms
binary_spectrograms_path = f"spectrograms/bin_spec_{window}_{threshold}.npz"
binary_spectrogram_data = np.load(binary_spectrograms_path, allow_pickle=True)
bin_specs = binary_spectrogram_data['spectrograms']

# Load full power spectrograms
full_power_spectrograms_path = f'spectrograms/power_spec_{station}_{window}_{threshold}.npz'
full_power_data = np.load(full_power_spectrograms_path, allow_pickle=True)
full_power_spectrograms = full_power_data['spectrograms']

# Reshape latent spaces for clustering
latent_spaces_reshaped = latent_spaces.reshape(latent_spaces.shape[0], -1)

# Apply K-means clustering on the latent spaces
n_clusters = clusters  # You can adjust the number of clusters based on your data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(latent_spaces_reshaped)

# Perform t-SNE on the latent spaces for visualization
tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init=init)
latent_tsne = tsne.fit_transform(latent_spaces_reshaped)

# Prepare the images for Bokeh
bin_spec_images = []
full_power_images = []
times = []  # To store times for cumulative plotting

for i, (binary_spec, full_power_spec) in enumerate(zip(bin_specs, full_power_spectrograms)):
    # Encode binary spectrogram
    img = Image.fromarray((binary_spec * 255).astype(np.uint8))
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

# Create color palet
colors = []
print(colors)
print(n_clusters)
if n_clusters <= 10:
    colors = [Category10[10][label] for label in cluster_labels]
elif n_clusters <= 20:
    colors = [Category20[20][label] for label in cluster_labels]

# Prepare data for Bokeh
source = ColumnDataSource(data=dict(
    x=latent_tsne[:, 0],
    y=latent_tsne[:, 1],
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

# Add hover tool with both images displayed
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
output_file(f"/fp/projects01/ec332/data/tsne_plots/file_no_res_{station}_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}_cluster_{perplexity}_{clusters}_mod.html")
save(p)

# Show the plot in a browser (if running locally)
show(p)

### Cumulative Time Plot ###
# Create a dictionary to store times for each cluster
cluster_times = defaultdict(list)

# Group times by cluster labels
for i, time in enumerate(times):
    cluster_times[cluster_labels[i]].append(time)

# Plot cumulative occurrences for each cluster
plt.figure(figsize=(10, 10))
for cluster_id, times in cluster_times.items():
    # Convert times to a pandas Series for cumulative counting
    time_series = pd.Series([pd.Timestamp(t) for t in times])
    time_series = time_series.sort_values()
    cumulative_counts = np.arange(1, len(time_series) + 1)
    
    plt.plot(time_series, cumulative_counts, label=f'Cluster {cluster_id}')

# Customize the plot
plt.title("Cumulative Occurrences of Each Cluster Over Time")
plt.xlabel("Time")
plt.ylabel("Cumulative Count")
plt.legend()
plt.grid(True)

# Format the x-axis to show dates clearly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()

# Save or show the cumulative time plot
plt.savefig(f"/fp/projects01/ec332/data/cumulative_cluster_plots/file_{station}_{window}_{threshold}_model_{num_epochs}_{weight}_{learning_rate}_{batch_size}_{bottle}_{clusters}.png")
plt.show()


