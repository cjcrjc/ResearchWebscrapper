import torch
import glob
import torch.nn as nn
from sortercnn import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Dataloader
#Path for training and testing directory
train_path='C:\\Users\\Cameron\\Desktop\\Coding\\Projects\Python\\ResearchWebscrapper\\Data Collection\\data\\train'
test_path='C:\\Users\\Cameron\\Desktop\\Coding\\Projects\Python\\ResearchWebscrapper\\Data Collection\\data\\test'

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32, shuffle=True
)

#Categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)
    
model=SorterCNN(num_classes=2).to(device)

#Optmizer and loss function
optimizer=Adam(model.parameters(),lr=0.01,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

#Calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/*/*.jpeg'))+len(glob.glob(train_path+'/*/*.jpg'))+len(glob.glob(train_path+'/*/*.png'))
test_count=len(glob.glob(test_path+'/*/*.jpeg'))+len(glob.glob(test_path+'/*/*.jpg'))+len(glob.glob(test_path+'/*/*.png'))
print(train_count,test_count)

num_epochs=3

#Model training and saving best model
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
        print(f'Epoch: {epoch+1}    Train Loss: {train_loss}     Train Accuracy: {train_accuracy}    Test Accuracy: {test_accuracy}')
        
        #Save the best model
        if test_accuracy>best_accuracy:
            torch.save(model.state_dict(),'best.model')
            best_accuracy=test_accuracy

#if more data is gathered do a final eval