import torch.nn as nn
from torchvision.transforms import transforms

resolution = 400

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
            nn.Conv2d(20, 32, 3,1,1),   # Conv3 (no padding)
            nn.BatchNorm2d(32),     # BatchNorm3
            nn.ReLU(),              # ReLU3
        )
        self.fc = nn.Linear(32 * int(resolution/2 * resolution/2), num_classes)
        self.iters = 0

    def forward(self, input):
        self.iters += 1
        print(self.iters)
        output = self.layers(input)
        #print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
    
#Transforms
transformer=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((resolution,resolution)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])