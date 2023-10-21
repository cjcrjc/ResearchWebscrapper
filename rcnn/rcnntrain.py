# Import necessary libraries
from PIL import Image
from torchvision.models.detection import *
from torchvision.transforms import ToTensor
import os, gc, glob, torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import date

# Get the directory path
dir_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check and set the device
print(device)

# Clean up memory and empty GPU cache
gc.collect()
torch.cuda.empty_cache()

# Define a custom dataset class
class CustData(torch.utils.data.Dataset):
    def __init__(self, df, unique_imgs, indices):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image_name = self.unique_imgs[self.indices[i]]
        boxes = self.df[self.df.image == image_name].values[:, 1:].astype("float")
        img = Image.open(image_name).convert("RGB")
        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
        target = {}
        target["boxes"] = torch.tensor(boxes)
        target["labels"] = labels
        return ToTensor()(img), target

# Define a custom collate function
def custom_collate(data):
    return data

# Create a Faster R-CNN model with a ResNet-50 backbone and two output classes
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)  # 2 for object and background

# Set the model to evaluation mode
model.eval()

# Read bounding box data from an Excel file
df = pd.read_excel('rcnn/bounding_boxes.xlsx')
unique_imgs = df.image.unique()
train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size=0.1)

# Define batch sizes and create data loaders for training and testing
train_batchsize = 2
train_loader = DataLoader(CustData(df, unique_imgs, train_inds), batch_size=train_batchsize, shuffle=True,
                          collate_fn=custom_collate, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(CustData(df, unique_imgs, val_inds), batch_size=int(train_batchsize / 2), shuffle=True,
                         collate_fn=custom_collate, pin_memory=True if torch.cuda.is_available() else False)

total_count = len(glob.glob(dir_path + '/train/*.*'))
test_count = int(total_count * 0.1)
train_count = int(total_count * 0.9)

# Define the optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()

# Specify the number of training epochs
num_epochs = 10
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for data in train_loader:
        imgs = []
        targets = []
        for d in data:
            imgs.append(d[0].to(device))
            targ = {}
            targ["boxes"] = d[1]["boxes"].to(device)
            targ["labels"] = d[1]["labels"].to(device)
            targets.append(targ)
        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values())
        print(f'Batch Loss: {loss.item()}')
        epoch_loss += loss.cpu().detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}    Train Loss: {epoch_loss}')

# Save the best model
torch.save(model.state_dict(), f'rcnn/{date.today()}.model')

# Evaluation on testing dataset
model.eval()
with torch.no_grad():
    test_accuracy = 0.0
    for data in test_loader:
        imgs = []
        for d in data:
            imgs.append(d[0].to(device))
        outputs = model(imgs)
