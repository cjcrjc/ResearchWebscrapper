import torch.nn as nn
from torchvision.transforms import transforms
from torch.nn.functional import *
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

resolution = 600

class RPN(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.cls_layer = nn.Conv2d(256, num_anchors, kernel_size=1)
        self.reg_layer = nn.Conv2d(256, 4 * num_anchors, kernel_size=1)

    def forward(self, x):
        x = relu(self.conv(x))
        class_scores = self.cls_layer(x)
        bbox_deltas = self.reg_layer(x)
        return class_scores, bbox_deltas


#CNN Sorter Model
class SorterCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SorterCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 12, 3,1,1),    # Conv1 (no padding)
            nn.BatchNorm2d(12),     # BatchNorm1
            nn.ReLU(),              # ReLU1
            nn.MaxPool2d(2),        # MaxPool
            nn.Conv2d(12, 20, 3,1,1),   # Conv2 (no padding)
            nn.ReLU(),              # ReLU2
            nn.MaxPool2d(2),        # MaxPool
            nn.Conv2d(20, 32, 3,1,1),   # Conv3 (no padding)
            nn.BatchNorm2d(32),     # BatchNorm3
            nn.ReLU(),              # ReLU3
        )
        self.fc = nn.Linear(32 * 150 * 150, num_classes)

    def forward(self, input):
        x = self.layers(input)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc(x)
        return x
    
#Transforms
transformer=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((resolution,resolution)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.5,0.5,0.5])
])