import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from network import LowDimNet, ControlNet
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

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
os.makedirs('control_models_swiglu', exist_ok=True)
# os.makedirs('control_plots_swiglu', exist_ok=True)

# Create a color map for all possible labels (0 to 9)
color_map = plt.cm.get_cmap('tab10', 10)

for num_classes in range(10, 11):
    print(f"Training a 2-Dimensional Autoencoder on MNIST with {num_classes} classes")

    plot_files = []
    # Select samples with labels in the range [0, num_classes)
    train_indices = torch.nonzero(train_dataset.targets < num_classes).squeeze()
    test_indices = torch.nonzero(test_dataset.targets < num_classes).squeeze()

    # Create subset datasets with selected samples
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = ControlNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Set the same initial weights for each run
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Training loop
    step = 0
    while True:
        if step >= num_steps:
            break
        for batch_idx, (data, targets) in enumerate(train_loader):
            if step >= num_steps:
                break
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # # Get the low-dimensional representations and predicted probabilities of the test set
            # low_dim_representations = []
            # labels = []
            # predicted_probs_all = []
            # with torch.no_grad():
            #     for images, target in test_loader:
            #         low_dim_rep = model.get_low_dim(images.to(device)).cpu().numpy()
            #         predicted_probs = model.predict_from_low_dim(torch.from_numpy(low_dim_rep).to(device)).cpu().detach().numpy()
            #         low_dim_representations.append(low_dim_rep)
            #         labels.append(target.numpy())
            #         predicted_probs_all.append(predicted_probs)

            # low_dim_representations = np.concatenate(low_dim_representations)
            # labels = np.concatenate(labels)
            # predicted_probs_all = np.concatenate(predicted_probs_all)

            # # Plot the low-dimensional representations
            # plt.figure(figsize=(8, 6))
            # for i in range(num_classes):
            #     mask = labels == i
            #     plt.scatter(low_dim_representations[mask, 0], low_dim_representations[mask, 1],
            #                 color=color_map(i), label=str(i), alpha=0.5)
            # plt.xlabel('Dimension 1')
            # plt.ylabel('Dimension 2')
            # plt.legend(title='Labels', loc='upper right')
            # plt.title(f'Training a 2-Dimensional Autoencoder on MNIST (Classes: {num_classes})')
            # # # don't show the axes
            # # plt.axis('off')
            # plt.tight_layout()

            # # show grid
            # plt.grid(True)

            # # Save the plot as a PNG file
            # plot_file = os.path.join('low_dim_plots_swiglu', f'step_{step}_classes_{num_classes}.png')
            # plt.savefig(plot_file)
            # plt.close()
            # plot_files.append(plot_file)

    # Evaluation on the test set after training
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy on the test set for {num_classes} classes: {accuracy:.2f}%")

    # Save the model after training
    model_file = os.path.join('control_models_swiglu', f'classes_{num_classes}.pth')
    torch.save(model.state_dict(), model_file)

    # # Create a GIF from the saved plot files
    # images = []
    # for plot_file in plot_files:
    #     images.append(imageio.imread(plot_file))

    # duration = 1 / 30
    # imageio.mimsave(f'gifs_swiglu/classes_{num_classes}.gif', images, duration=duration)