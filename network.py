import torch
import torch.nn as nn
import torch.nn.functional as F

class ControlNet(nn.Module):
    def __init__(self, num_classes):
        super(ControlNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.gate1 = nn.Linear(28 * 28, 64)
        self.gate2 = nn.Linear(64, 64)
        self.gate3 = nn.Linear(64, 64)
    
    def swiglu(self, x, gate):
        return x * torch.sigmoid(gate)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.swiglu(self.fc1(x), self.gate1(x))
        x = self.swiglu(self.fc2(x), self.gate2(x)) + x
        x = self.swiglu(self.fc3(x), self.gate3(x)) + x
        x = self.fc4(x)
        return x
    
    def get_representation(self, x, layer):
        x = x.view(-1, 28 * 28)
        x = self.swiglu(self.fc1(x), self.gate1(x))
        if layer == 1:
            return x
        x = self.swiglu(self.fc2(x), self.gate2(x)) + x
        if layer == 2:
            return x
        x = self.swiglu(self.fc3(x), self.gate3(x)) + x
        return x
    
    def predict_from_representation(self, x, layer):
        if layer == 1:
            x = self.swiglu(self.fc2(x), self.gate2(x)) + x
            x = self.swiglu(self.fc3(x), self.gate3(x)) + x
        elif layer == 2:
            x = self.swiglu(self.fc3(x), self.gate3(x)) + x
        x = self.fc4(x)
        return x

class LowDimNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LowDimNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 2)
        self.fc3 = nn.Linear(2, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.gate1 = nn.Linear(28 * 28, 64)
        self.gate2 = nn.Linear(64, 2)  # Added gate for fc2
        self.gate3 = nn.Linear(2, 64)

    def swiglu(self, x, gate):
        return x * torch.sigmoid(gate)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.swiglu(self.fc1(x), self.gate1(x))
        x = self.swiglu(self.fc2(x), self.gate2(x))  # Added swiglu activation for fc2
        x = self.swiglu(self.fc3(x), self.gate3(x))
        x = self.fc4(x)
        return x

    def get_low_dim(self, x):
        x = x.view(-1, 28 * 28)
        x = self.swiglu(self.fc1(x), self.gate1(x))
        x = self.swiglu(self.fc2(x), self.gate2(x))  # Added swiglu activation for fc2
        return x

    def predict_from_low_dim(self, x):
        x = self.swiglu(self.fc3(x), self.gate3(x))
        x = self.fc4(x)
        return x