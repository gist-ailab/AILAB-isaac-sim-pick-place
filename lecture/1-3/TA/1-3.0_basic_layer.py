# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-3.0 Basic Network Layer
# ---- ---- ---- ----


import torch
import torch.nn as nn


fc_layer = nn.Linear(7*7*32, 1)
conv_layer = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),


sequential_layer1 = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
