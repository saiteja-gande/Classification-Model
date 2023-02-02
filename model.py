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
        
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model')

