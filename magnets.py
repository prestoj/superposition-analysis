import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from network import LowDimNet
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import math

# # Set random seeds for reproducibility
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_steps = 1000

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# Create directories to store the model save files and plots
os.makedirs('low_dim_models_swiglu', exist_ok=True)
os.makedirs('low_dim_plots_swiglu', exist_ok=True)

# Create a color map for all possible labels (0 to 9)
color_map = plt.cm.get_cmap('tab10', 10)
num_classes = 5

train_indices = torch.nonzero(train_dataset.targets < num_classes).squeeze()
test_indices = torch.nonzero(test_dataset.targets < num_classes).squeeze()

# Create subset datasets with selected samples
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

model = LowDimNet(num_classes).to(device)
model.load_state_dict(torch.load(f'low_dim_models_swiglu/classes_{num_classes}.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

# Get the low-dimensional representations and predicted probabilities of the test set
low_dim_representations = []
labels = []
predicted_probs_all = []
with torch.no_grad():
    for images, target in test_loader:
        low_dim_rep = model.get_low_dim(images.to(device)).cpu().numpy()
        predicted_probs = model.predict_from_low_dim(torch.from_numpy(low_dim_rep).to(device)).cpu().detach().numpy()
        low_dim_representations.append(low_dim_rep)
        labels.append(target.numpy())
        predicted_probs_all.append(predicted_probs)

low_dim_representations = np.concatenate(low_dim_representations) # (10000, 2)
labels = np.concatenate(labels) # (10000,)
predicted_probs_all = np.concatenate(predicted_probs_all) # (10000, 10)

def generate_vectors(n):
    angle = 2 * torch.pi / n
    vectors = []
    for i in range(n):
        x = torch.cos(torch.tensor(i, dtype=torch.float32) * angle)
        y = torch.sin(torch.tensor(i, dtype=torch.float32) * angle)
        vectors.append(torch.stack([x, y]))
    return torch.stack(vectors)

num_vectors = 4
# for num_vectors in range(1, 20):
# # Generate initial vectors
# vectors_tensor = generate_vectors(num_vectors)
# generate random vectors
vectors_tensor = torch.randn(num_vectors, 2)
vectors_tensor = vectors_tensor / torch.norm(vectors_tensor, dim=1, keepdim=True)
vectors_tensor.requires_grad = True
samples = torch.tensor(low_dim_representations)
samples = samples / torch.norm(samples, dim=1, keepdim=True)

optimizer = optim.Adam([vectors_tensor], lr=1e-1)

for i_step in range(10000):

    # finding the great circle distance on the n-sphere would be ideal but the math seems a tad complicated. will revist
    # for now, we will just use the euclidean distance

    sample_distances = torch.cdist(vectors_tensor, samples, p=2) # (num_vectors, num_samples)
    vector_distances = torch.cdist(vectors_tensor, vectors_tensor, p=2) # (num_vectors, num_vectors)
    # print(sample_distances[:, 0])

    # loss is the mean of the distances between the vectors and the samples minus the distances between the vectors
    # loss = 0.5 * (0.01 * vector_distances.mean() - 1 * sample_distances.mean())
    sample_distance_loss = torch.min(sample_distances, dim=0).values.mean()
    vector_distance_loss = vector_distances.mean()
    loss = sample_distance_loss - 0.01 * vector_distance_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # normalize the vectors
    vectors_tensor.data = vectors_tensor.data / torch.norm(vectors_tensor.data, dim=1, keepdim=True)

print(f"Num Vectors {num_vectors}, Loss: {loss.item()}, Sample Distance Loss: {sample_distance_loss.item()}, Vector Distance Loss: {vector_distance_loss.item()}")

optimized_vectors = vectors_tensor.detach().numpy()

# Convert low-dimensional representations to angular coordinates (theta)
theta = np.arctan2(low_dim_representations[:, 1], low_dim_representations[:, 0])

# Convert angular coordinates to Cartesian coordinates on the unit circle
x = np.cos(theta)
y = np.sin(theta)

# Create a plot
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the unit circle
circle = plt.Circle((0, 0), 1, fill=False)
ax.add_artist(circle)

# Plot the low-dimensional representations on the unit circle
ax.scatter(x, y, c=labels, cmap=color_map, alpha=0.01)

# Plot the optimized vectors
for i in range(num_vectors):
    vec = optimized_vectors[i]
    vec_norm = vec / np.linalg.norm(vec)
    ax.arrow(0, 0, vec_norm[0], vec_norm[1], head_width=0.05, head_length=0.1, fc='k', ec='k')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Low-Dimensional Representations on Unit Circle and Optimized Vectors')

# Set equal aspect ratio for better visualization
ax.set_aspect('equal')

# Set the limits of the plot
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.show()