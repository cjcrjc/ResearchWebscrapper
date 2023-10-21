import torch.nn as nn
from torchvision.transforms import transforms
from torch.nn.functional import *
import os

# Get the path of the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define the resolution for image transformation
resolution = 600

# Define a Convolutional Neural Network (CNN) model for sorting
class SorterCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SorterCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1, 1),    # Conv1 (no padding)
            nn.BatchNorm2d(12),           # BatchNorm1
            nn.ReLU(),                   # ReLU1
            nn.MaxPool2d(2),             # MaxPool
            nn.Conv2d(12, 20, 3, 1, 1),  # Conv2 (no padding)
            nn.ReLU(),                   # ReLU2
            nn.MaxPool2d(2),             # MaxPool
            nn.Conv2d(20, 32, 3, 1, 1),  # Conv3 (no padding)
            nn.BatchNorm2d(32),           # BatchNorm3
            nn.ReLU()                    # ReLU3
        )
        self.fc = nn.Linear(32 * int(resolution / 4) ** 2, num_classes)

    def forward(self, input):
        x = self.layers(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define a series of image transformations
transformer = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert image to grayscale with 3 channels
    transforms.Resize((resolution, resolution)),  # Resize the image to the specified resolution
    transforms.RandomHorizontalFlip(),  # Apply random horizontal flip
    transforms.ToTensor(),  # Convert image to tensors
    transforms.Normalize([0.5, 0.5, 0.5],  # Normalize the image data
                         [0.5, 0.5, 0.5])
])
