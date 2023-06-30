# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-3.0 Basic Convolution Network Structure
# ---- ---- ---- ----


import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Linear model (width*height*channel of the last feature map, Number of class)
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        flatten = out.view(out.size(0), -1)  # Flatten
        # flatten = out.reshape(out.size(0), -1) # We can also use '.reshape'
        score = self.fc(flatten) # Score
        # prob = F.softmax(score) 
        '''If you use 'torch.nn.CrossEntropyLoss', you don't need to add 'F.softmax'.'''
        '''Basically 'torch.nn.CrossEntropyLoss' contains log softmax'''
        return score