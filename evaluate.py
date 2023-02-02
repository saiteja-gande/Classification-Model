import torch
from torch.utils.data import DataLoader
from model import Network
from utils import TestSet, evaluate

# dataset location
path = 'dataset'

model = Network().eval()
model.load_state_dict(torch.load('model'))

loader = DataLoader(TestSet(path), batch_size=1)

print(evaluate(model, loader))