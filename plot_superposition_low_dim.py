from network import ControlNet, LowDimNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import imageio

# Load the MNIST dataset
batch_size = 64
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create a directory to store the plots
os.makedirs('low_dim_plots', exist_ok=True)

# Get the list of saved model files sorted by epoch and batch
model_files = sorted([file for file in os.listdir('low_dim_models') if file.endswith('.pth')],
                     key=lambda x: (int(x.split('_')[1]), int(x.split('_')[3].split('.')[0])))

# Iterate through each saved model
plot_files = []
for file in model_files:
    # Load the trained model
    low_dim_model = LowDimNet()
    low_dim_model.load_state_dict(torch.load(os.path.join('low_dim_models', file)))
    low_dim_model.eval()

    # Get the low-dimensional representations and predicted probabilities of the test set
    low_dim_representations = []
    labels = []
    predicted_probs_all = []
    with torch.no_grad():
        for images, target in test_loader:
            low_dim_rep = low_dim_model.get_low_dim(images)
            predicted_probs = low_dim_model.predict_from_low_dim(low_dim_rep).detach().numpy()
            low_dim_representations.append(low_dim_rep)
            labels.append(target)
            predicted_probs_all.append(predicted_probs)

    low_dim_representations = torch.cat(low_dim_representations).numpy()
    labels = torch.cat(labels).numpy()
    predicted_probs_all = np.concatenate(predicted_probs_all)

    # Create a color map
    color_map = plt.cm.get_cmap('tab10', 10)

    # Plot the low-dimensional representations
    plt.figure(figsize=(8, 6))
    for i in range(10):
        mask = labels == i
        plt.scatter(low_dim_representations[mask, 0], low_dim_representations[mask, 1],
                    c=color_map(i), label=str(i), alpha=0.5)

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(title='Labels', loc='upper right')
    plt.title(f'Training a 2-Dimensional Autoencoder on MNIST')
    # don't show the axes
    plt.axis('off')
    plt.tight_layout()

    # Save the plot as a PNG file
    plot_file = os.path.join('low_dim_plots', f'{file[:-4]}.png')
    plt.savefig(plot_file)
    plt.close()

    plot_files.append(plot_file)

# Create a GIF from the saved plot files
images = []
for plot_file in plot_files:
    images.append(imageio.imread(plot_file))

# duration = 30 / len(images)
duration = 1 / 30

imageio.mimsave('low_dim_representations.gif', images, duration=duration)