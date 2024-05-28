import os
import imageio

# Directory containing the plot files
plot_dir = 'low_dim_plots'

# Get the list of plot files sorted by epoch and batch
plot_files = sorted([file for file in os.listdir(plot_dir) if file.endswith('.png')],
                    key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3].split('.')[0])))

# Create a GIF from the saved plot files
images = []
for plot_file in plot_files:
    images.append(imageio.imread(os.path.join(plot_dir, plot_file)))

duration = 1 / 30

imageio.mimsave('low_dim_representations.gif', images, duration=duration)