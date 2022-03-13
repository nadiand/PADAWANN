# General imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class InitialModel(nn.Module):
    """
    A class defining the structure of the PyTorch models used in the GA.
    To use a different structure, simply change the __init__ and forward methods.
    NB: If you plan to apply transfer learning on the model (retraining its last
    layer), do not change the last layer's name "fc2".

    This architecture in particular is made specifically for an MNIST classifier
    and is taken from https://github.com/pytorch/examples/blob/master/mnist/main.py.
    """
    def __init__(self):
        super(InitialModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Just a feed-forward architecture, with no convolution
# Inspired by: https://www.youtube.com/watch?v=O5xeyoRL95U
# class InitialModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc0 = nn.Flatten()
#         self.fc1 = nn.Linear(784, 64)
#         self.fc2 = nn.Linear(64, 10)
    
#     def forward(self, x):
#         x = self.fc0(x)
#         x = F.relu(x)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output