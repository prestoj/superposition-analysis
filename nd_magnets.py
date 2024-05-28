import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from network import ControlNet
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_steps = 1000
layers = [1, 2, 3]

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

num_classes = 10

train_indices = torch.nonzero(train_dataset.targets < num_classes).squeeze()
test_indices = torch.nonzero(test_dataset.targets < num_classes).squeeze()

# Create subset datasets with selected samples
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

model = ControlNet(num_classes).to(device)
model.load_state_dict(torch.load(f'control_models_swiglu/classes_{num_classes}.pth'))

# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

for layer in layers:
    raw_images = []
    representations = []
    labels = []
    with torch.no_grad():
        for images, target in test_loader:
            raw_images.append(images)
            rep = model.get_representation(images.to(device), layer=layer).cpu().numpy()
            representations.append(rep)
            labels.append(target.numpy())

    low_dim_representations = np.concatenate(representations) # (10000, 64)
    labels = np.concatenate(labels) # (10000,)

    num_vectors = 1024
    vector_dim = 64
    vectors_tensor = torch.randn(num_vectors, vector_dim)
    vectors_tensor = vectors_tensor / torch.norm(vectors_tensor, dim=1, keepdim=True)
    vectors_tensor.requires_grad = True
    samples = torch.tensor(low_dim_representations)
    samples = samples / torch.norm(samples, dim=1, keepdim=True)

    optimizer = optim.Adam([vectors_tensor], lr=1e-3)

    for i_step in range(1000):
        # finding the great circle distance on the n-sphere would be ideal but the math seems a tad complicated. will revist
        # for now, we will just use the euclidean distance

        sample_distances = torch.cdist(vectors_tensor, samples, p=2) # (num_vectors, num_samples)
        vector_distances = torch.cdist(vectors_tensor, vectors_tensor, p=2) # (num_vectors, num_vectors)

        sample_distance_loss = torch.min(sample_distances, dim=0).values.mean()
        vector_distance_loss = vector_distances.mean()
        loss = sample_distance_loss - 1 * vector_distance_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # normalize the vectors
        vectors_tensor.data = vectors_tensor.data / torch.norm(vectors_tensor.data, dim=1, keepdim=True)

        if i_step % 100 == 0:
            print(f"Step {i_step}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}, Vector Distance Loss: {vector_distance_loss.item()}")

    print(f"Num Vectors {num_vectors}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}, Vector Distance Loss: {vector_distance_loss.item()}")

    optimized_vectors = vectors_tensor.detach().numpy()

    # Create the directory if it doesn't exist
    os.makedirs(f"magnet/control_10/layer_{layer}", exist_ok=True)

    # find all the samples that are closest to each vector
    dead_vectors = 0
    for i_vector in range(num_vectors):
        closest_indices = (sample_distances.argmin(dim=0) == i_vector).nonzero().squeeze()
        if closest_indices.numel() > 0:
            if closest_indices.dim() == 0:
                closest_indices = closest_indices.unsqueeze(0)
            closest_indices = closest_indices.tolist()
            # print(i_vector, labels[closest_indices])

            # Plot the closest samples in a grid and save the figure
            num_samples = len(closest_indices)
            num_cols = min(10, num_samples)
            num_rows = math.ceil(num_samples / num_cols)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*1.5, num_rows*1.5))

            if num_rows == 1 and num_cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, idx in enumerate(closest_indices):
                img = raw_images[idx // batch_size][idx % batch_size].squeeze().numpy()
                axes[i].imshow(img, cmap='gray')
                axes[i].axis('off')

            for j in range(num_samples, num_rows * num_cols):
                axes[j].axis('off')

            plt.tight_layout()
            plt.savefig(f"magnet/control_10/layer_{layer}/vector_{i_vector}.png")
            plt.close(fig)
        else:
            # print(i_vector, "No samples closest to this vector")
            dead_vectors += 1

    print(dead_vectors, num_vectors)