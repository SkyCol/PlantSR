import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import math


class SRCNN(nn.Module):
    def __init__(self, nChannel=3):
        super(SRCNN,self).__init__()
        self.conv1 = nn.Conv2d(nChannel, 64,
            kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32,
            kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=nChannel,
            kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x