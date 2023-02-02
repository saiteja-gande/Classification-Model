#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset
import torch
import torchvision
import os
from PIL import Image
import natsort
#from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models, transforms


class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
        self.training = training
        self.path = path
        
        #For training data
        self.transform1 = transforms.Compose([
             transforms.Resize((224, 224)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomRotation(degrees=10),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        #For validation data
        self.transform2 = transforms.Compose([
             transforms.Resize((224,224)),
             transforms.ToTensor(),
             transforms.Normalize(mean= ([0.485, 0.456, 0.406]),std = ([0.229, 0.224, 0.225]))
             ])
         
        if self.training == True:
            self.dataset = ImageFolder(path + './train',transform=self.transform1)
        else:
            self.path = path
            self.sorted_image = natsort.natsorted(os.listdir(self.path))
    
    def __len__(self):
        return len(self.dataset)
            
    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        if self.training == True:
            image,label = self.dataset[index]
            return image,label
        
        else:
            img = os.path.join(self.path,self.sorted_image[index])
            
            image = self.transform2(Image.open(img).convert("RGB"))
            return (image, )
        raise NotImplementedError


# In[2]:


import torch
from torch.utils.data import Dataset
import csv
import pandas as pd
# from data import ChristmasImages


class TestSet(Dataset):
    
    def __init__(self, path):
        super().__init__()
        #change location to the path of the validation dataset
        self.dataset = ChristmasImages(path + 'location', training=False)
        
        with open(path + '/val.csv') as file:
            reader = csv.reader(file)
            next(reader)
            labels = {}
            for row in reader:
                labels[int(row[0])] = int(row[1])
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        label = self.labels[idx]
        return image, label



# In[3]:


import torch
import torch.nn as nn
import torchvision.models as models


class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    # load the ResNet-18 model
        self.model = models.resnet18(pretrained=True)

        # replace the final fully-connected layer with a new linear layer
        #self.num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear((self.model.fc.in_features), 8)

        # freeze all layers
        for param in self.model.parameters():
            param.requiresGrad = False
        for param in self.model.fc.parameters():
            param.requiresGrad = True
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################

        x = self.model(x)
        return x
        
    
#     def save_model(self):
        
#         #############################
#         # Saving the model's weitghts
#         # Upload 'model' as part of
#         # your submission
#         # Do not modify this function
#         #############################
        
#         torch.save(self.state_dict(), 'model')



# In[4]:


from path import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
path = Path("path") # path to your train folder
train_ds = ChristmasImages(path)
train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)


# In[8]:


print(len(train_ds))


# In[ ]:


model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


## YOUR CODE HERE ##
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# In[ ]:


def train(model, train_loader, val_loader=None,  epochs=30, save=True):
    best_acc = 0
    total= train_acc = correct = 0
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0 ):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        train_acc = 100. * correct / total
        print('train accuracy: %d %%' % train_acc)
        
        
        if save:
            if train_acc > best_acc:
                best_acc = train_acc
                torch.save(model.state_dict(), "save-"+str(epoch))

# In[ ]:



train(model, train_loader,save=True)


# In[ ]:




# In[ ]:






# In[ ]:




