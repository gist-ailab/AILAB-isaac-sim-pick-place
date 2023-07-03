# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-4.0 Introduction to Object Detection
# ---- ---- ---- ----


import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import cv2 
import numpy as np
from PIL import Image


model  = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # download and load pre-trained checkpoints


# Show layers of Faster-RCNN
model


device = "cuda"
model = model.to(device)
model.eval()
## Load any images
image_path = "//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/coco_val/COCO_val2014_000000000073.jpg"
## Convert images to tensor
test_image = Image.open(image_path)


# Visualize image
show_image = Image.open(image_path)
show_image


to_tensor = torchvision.transforms.ToTensor()

### Convert PIL image to Torch Tensor
test_image = to_tensor(test_image) 
print("Shape of tensor :", test_image.shape) 
print(test_image)
# B, C, H,W
### Unsqueeze the tensor by axis 0
test_image = test_image.unsqueeze(0)
print("Shape of tensor after unsqueeze :",test_image.shape)
print(test_image)


predictions = model(test_image.to(device))

predictions

print('Type of output :', type(predictions))

print('Type of element of the output :', type(predictions[0]))

print('Keys of element of the output :', list(predictions[0].keys()))


print('Type of the element of "boxes" : ', type(predictions[0]['boxes']))

print('Length of the elements of "boxes" : ', len(predictions[0]['boxes']))

print('Length of the elements of "labels" : ', len(predictions[0]['labels']))

print('Length of the elements of "scores" : ', len(predictions[0]['scores']))


def overlay_instances(img_path, predictions, threshold=0.8):
    ori_img = cv2.imread(img_path) # load original image 

    for idx in range(len(predictions[0]['boxes'])): # draw all predicted instances over original image
        score = predictions[0]['scores'][idx].cpu().detach().numpy().item() # confidence score of each prediction 
        if score < threshold: 
            continue # if the confidence score is lower than threshold, do not draw the instance 
        box = x1, y1, x2, y2 = predictions[0]['boxes'][idx].cpu().detach().numpy() 
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255,0,0),2) # draw rectangle on image, [ params : (np.array of image), (x1, y1), (x2, y2), color, thickness]
    return Image.fromarray(ori_img)


overlay_instances("//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/coco_val/COCO_val2014_000000000073.jpg", predictions, 0.9)


# Get file names under the specific folder.
import os
os.listdir("//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/coco_val") 
# Returns list of file names in the folder "//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/coco_val"


image_path = "//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/coco_val/COCO_val2014_000000000133.jpg"

# Load image using PIL.
test_image = Image.open(image_path) 

# Convert PIL image to Tensor 
to_tensor = torchvision.transforms.ToTensor()

test_image = to_tensor(test_image) 

# Unsqueeze Tensor by axis 0 
test_image = test_image.unsqueeze(0)

# Inference on the model 
predictions = model(test_image.to(device))

# Print Inference Result
print(predictions)

# Visualize result

overlay_instances(image_path, predictions, 0.5)




print()