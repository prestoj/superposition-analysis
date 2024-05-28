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

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
num_classes = 6

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

def rotate_vectors(vectors, rotation_angle):
    cos_angle = torch.cos(rotation_angle)
    sin_angle = torch.sin(rotation_angle)
    rotation_matrix = torch.stack([
        torch.stack([cos_angle, -sin_angle]),
        torch.stack([sin_angle, cos_angle])
    ])
    return torch.matmul(vectors, rotation_matrix)

num_vectors = num_classes

# Generate initial vectors
vectors_tensor = generate_vectors(num_vectors)

# Define the rotation angle as a learnable parameter
rotation_angle = nn.Parameter(torch.tensor(0.0))

# Define the optimization criterion
criterion = nn.CosineSimilarity(dim=2)

# Create an optimizer for the rotation angle
optimizer = optim.Adam([rotation_angle], lr=1e-2)

num_epochs = 1000

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Optimize the rotation angle
for epoch in range(num_epochs):
    # Rotate the vectors based on the current rotation angle
    rotated_vectors = rotate_vectors(vectors_tensor, rotation_angle)

    # Convert low_dim_representations to a tensor
    low_dim_representations_tensor = torch.tensor(low_dim_representations, dtype=torch.float32)

    # Reshape low_dim_representations_tensor and rotated_vectors
    low_dim_representations_tensor = low_dim_representations_tensor.unsqueeze(1)  # (10000, 1, 2)
    rotated_vectors = rotated_vectors.unsqueeze(0)  # (1, 10, 2)

    # Compute the cosine similarities between each point and each rotated vector
    cosine_similarities = criterion(low_dim_representations_tensor, rotated_vectors)

    # Find the maximum cosine similarity for each point
    max_cosine_similarities, _ = torch.max(cosine_similarities, dim=1)

    # Compute the loss as the negative mean of the maximum cosine similarities
    loss = -torch.mean(max_cosine_similarities)
    # print(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Rotation Angle: {rotation_angle.item():.4f}")

# Retrieve the optimized rotation angle
optimized_rotation_angle = rotation_angle.item()

# Rotate the vectors with the optimized rotation angle
optimized_vectors = rotate_vectors(vectors_tensor, torch.tensor(optimized_rotation_angle)).numpy()
print(optimized_vectors)

# Plot the low-dimensional representations and the rotated vectors
plt.figure(figsize=(8, 6))
plt.scatter(low_dim_representations[:, 0], low_dim_representations[:, 1], c=labels, cmap=color_map, alpha=0.5)
for i in range(num_vectors):
    plt.arrow(0, 0, optimized_vectors[i, 0], optimized_vectors[i, 1], head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title(f'Low-Dimensional Representations and Optimized Vectors')
plt.show()

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