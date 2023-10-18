import torch
import glob
import torch.nn as nn
from sortercnn import *
from torch.optim import Adam
from torch.autograd import Variable
import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from dataloader import *
from datetime import date

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

#Calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/*/*.*'))
test_count=len(glob.glob(test_path+'/*/*.*'))
print(f"Training file count: {train_count}, Testing file count: {test_count}")
    
model=SorterCNN(num_classes=2).to(device)

#Optmizer and loss function
optimizer=Adam(model.parameters(),lr=0.01,weight_decay=0.0001)
dataset = torchvision.datasets.ImageFolder(train_path,transform=transformer)
class_count = [0] * len(dataset.classes)
for i, (image_path, label) in enumerate(dataset.samples):
    class_count[label] += 1
class_weights = torch.Tensor(list(map(float,class_count / np.linalg.norm(class_count))))
loss_function=nn.CrossEntropyLoss(weight=class_weights)

#Show Images
if not True:
    imgs = [img for (img,label) in test_loader]
    #print(imgs[0].numpy().shape)
    for i in range(len(imgs[0])):
        print(np.transpose(imgs[0][i].cpu().detach().numpy()))
        plt.imshow(np.transpose(255*imgs[0][i].cpu().detach().numpy(), (1, 2, 0)).astype(np.uint8))
        plt.show()

#Use oversampled Data?
oversample = False
if oversample:
    train_loader = oversampled_train_loader

#Model training and saving best model
num_epochs = 10
best_accuracy=0.0
for epoch in range(num_epochs):
    model.iters = 0
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        optimizer.zero_grad()

        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
        if i % 5 == 0:
            print(f"TRAIN: Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}")
        
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    
    # Evaluation on testing dataset
    model.eval()
    with torch.no_grad():
        test_accuracy=0.0
        for i, (images,labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())
                
            outputs=model(images)
            _,prediction=torch.max(outputs.data,1)
            test_accuracy+=int(torch.sum(prediction==labels.data))
            
        test_accuracy=test_accuracy/test_count
        print(f'TEST: Epoch: {epoch+1}    Train Loss: {train_loss}     Train Accuracy: {train_accuracy}    Test Accuracy: {test_accuracy}')
        
        #Save the best model
        if test_accuracy>best_accuracy:
            torch.save(model.state_dict(),f'1. Data Collection/{date.today()}-accuracy:{test_accuracy}.model')
            best_accuracy=test_accuracy