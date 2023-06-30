# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-3.0 Basic FC Network Structure
# ---- ---- ---- ----


import torch.nn as nn


class BasicNet(nn.Module):
    def __init__(self, output_shape=10):
        super(BasicNet, self).__init__()
        self.fc = nn.Linear(32*32*3, output_shape)

    def forward(self, x):
        out = self.fc(x)

        return out


class FCNet(nn.Module):                                         #
    def __init__(self, num_classes=10):                         #
        super(FCNet, self).__init__()                           #

        self.fc0 = nn.Linear(32*32*3, 256)                      #
        self.fc1 = nn.Linear(256, 128)                          #
        self.fc2 = nn.Linear(128, 64)                           #
        self.fc3 = nn.Linear(64, 32)                            #
        self.fc4 = nn.Linear(32, num_classes)                   #

    def forward(self, x):                                       #
        
        out0 = self.fc0(x)                                      #
        out1 = self.fc1(out0)                                   #
        out2 = self.fc2(out1)                                   #
        out3 = self.fc3(out2)                                   #
        out4 = self.fc4(out3)                                   #

        score = self.fc(out4) # Score                           #
        # prob = F.softmax(score) 
        '''If you use 'torch.nn.CrossEntropyLoss', you don't need to add 'F.softmax'.'''
        '''Basically 'torch.nn.CrossEntropyLoss' contains log softmax'''
        return score                                            #

