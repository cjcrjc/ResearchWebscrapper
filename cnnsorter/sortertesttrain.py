import torch, glob, shutil
import torch.nn as nn
from sortercnn import *
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

# Unzip Data
data_path = os.path.join(dir_path, "data")
if os.path.isfile(data_path + ".zip"):
    shutil.unpack_archive(data_path + ".zip", data_path)

# Check for device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define paths for training and testing data
train_path = dir_path + '/data/train'
test_path = dir_path + '/data/test'

# Create data loaders for training and testing datasets
train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=16,
    shuffle=True
)

# Categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

# Calculate the size of training and testing datasets
train_count = len(glob.glob(train_path + '/*/*.*'))
test_count = len(glob.glob(test_path + '/*/*.*'))
print(f"Training file count: {train_count}, Testing file count: {test_count}")

# Create an instance of the SorterCNN model
model = SorterCNN(num_classes=2).to(device)

# Define optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

# Calculate class weights based on the distribution of classes in the training data
dataset = torchvision.datasets.ImageFolder(train_path, transform=transformer)
class_count = [0] * len(dataset.classes)
for i, (image_path, label) in enumerate(dataset.samples):
    class_count[label] += 1
class_weights = torch.Tensor(list(map(float, class_count / np.linalg.norm(class_count))))
loss_function = nn.CrossEntropyLoss(weight=class_weights).to(device)

# Show Images (conditionally)
if not True:
    imgs = [img for (img, label) in test_loader]
    for i in range(len(imgs[0])):
        plt.imshow(np.transpose(255 * imgs[0][i].cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        plt.show()

# Model training and saving the best model
num_epochs = 10
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.iters = 0

    # Evaluation and training on the training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(images.to(device))
        loss = loss_function(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        train_accuracy += int(torch.sum(prediction.cpu() == labels.data))

        if i % 5 == 0:
            print(f"TRAIN: Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on the testing dataset
    model.eval()
    with torch.no_grad():
        test_accuracy = 0.0
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            test_accuracy += int(torch.sum(prediction == labels.data))

        test_accuracy = test_accuracy / test_count
        print(f'TEST: Epoch: {epoch + 1}    Train Loss: {train_loss}     Train Accuracy: {train_accuracy}    Test Accuracy: {test_accuracy}')

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

# Save the trained model with the date and accuracy in the file name
torch.save(model.state_dict(), f'cnnsorter/{date.today()}-accuracy-{int(100 * best_accuracy)}.model')

# Zip data and remove the uncompressed folder
shutil.make_archive(data_path, 'zip', data_path)
shutil.rmtree(data_path)
