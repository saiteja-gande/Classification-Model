import torch
from torch.utils.data import Dataset
import csv
import pandas as pd
from data import ChristmasImages


class TestSet(Dataset):
    
    def __init__(self, path):
        super().__init__()
        self.dataset = ChristmasImages(path + '/data/val', training=False)
        
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


def evaluate(model, loader):
    model = model.cuda()
    indices = []
    index = 0
    predictions=[]
    accuracy = 0.
    with torch.no_grad():
        for image, label in loader:
            image, label = image.cuda(), label.cuda()
            indices.append(index)
            _, prediction = model(image).max(dim=1)
            predictions.append(prediction.item())
            df= pd.DataFrame({'Id': indices , 'Category' : predictions})
            df.to_csv('predictions.csv', index=False)
            accuracy += (prediction == label).sum().item()
            index = index + 1

    accuracy /= len(loader)
    return accuracy
