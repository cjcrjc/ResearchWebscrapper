from sortercnn import *
import torchvision, math
from torch.utils.data import DataLoader, Dataset

#Dataloader
#Path for training and testing directory
train_path=dir_path+'/data/train'
test_path=dir_path+'/data/test'

# Define a custom dataset that oversamples the minority class
class OversampledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = torchvision.datasets.ImageFolder(root, transform=transform)
        self.class_count = [0] * len(self.dataset.classes)
        self.indices = []
        
        # Collect indices of each class
        for i, (image_path, label) in enumerate(self.dataset.samples):
            self.indices.append(i)
            self.class_count[label] += 1
        
        # Calculate the target count as the maximum class count
        self.target_count = max(self.class_count)
        
        # Initialize an empty list to store oversampled indices
        self.oversampled_indices = []
        
        # Populate oversampled_indices to balance the class distribution
        for idx in self.indices:
            label = self.dataset.samples[idx][1]
            if self.class_count[label] < self.target_count:
                self.oversampled_indices.append(idx)
        self.oversampled_indices = self.oversampled_indices * math.ceil((self.target_count/len(self.oversampled_indices)))
        while len(self.oversampled_indices) > self.target_count:
            self.oversampled_indices.pop()
        #print(self.dataset.classes,self.class_count, self.oversampled_indices)

    def __getitem__(self, index):
        # Use oversampled indices for data retrieval
        print(self.dataset[self.oversampled_indices[index]])
        return self.dataset[self.oversampled_indices[index]]
    
    def __len__(self):
        return self.target_count * len(self.dataset.classes)

# Create oversampled training dataset
oversampled_train_dataset = OversampledDataset(train_path, transform=transformer)

# Data loader for oversampled training dataset
oversampled_train_loader = DataLoader(
    oversampled_train_dataset,
    batch_size=64,
    shuffle=True
)

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=64, shuffle=True
)

# Data loader for the test dataset
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=32,
    shuffle=True
)
