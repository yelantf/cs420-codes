import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
import numpy as np

class conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, momentum=0.1):
        super(conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=momentum, affine=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class fully_connected(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(fully_connected, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x