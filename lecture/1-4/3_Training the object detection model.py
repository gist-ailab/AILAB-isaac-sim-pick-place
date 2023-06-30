# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-4.0 Train Object Detection Model
# ---- ---- ---- ----


import os 
from PIL import Image 
import matplotlib.pyplot as plt
import cv2

data_root = '//content/drive/MyDrive/coding_lecture/05.19.CodingLecture7/PennFudanPed/'
# /content/drive/MyDrive/coding_lecture/05.19.CodingLecture7/PennFudanPed
png_root = os.path.join(data_root, 'PNGImages')  # concatenate strings
mask_root = os.path.join(data_root, 'PedMasks')

png_filename = os.path.join(png_root, 'FudanPed00001.png')
mask_filename = os.path.join(mask_root, 'FudanPed00001_mask.png') 

print(png_filename)
print(mask_filename)


png_img = Image.open(png_filename) 
# Show image
png_img


import numpy as np
mask_img = np.array(Image.open(mask_filename))

print('Shape of mask :', mask_img.shape)
plt.axis('off')
plt.imshow(mask_img, cmap='gray')


print(np.unique(mask_img)) # return unique value of numpy array
# 0 for background, others for the objects


''' print unique value in image '''
for i in np.unique(mask_img):
    print('Unique value : ', i)
    x, y = np.where(mask_img == i) # return x, y coordinates of pixel whose value is "i" 
    print(' Y coordinates of pixel whose value is %d : '%(i), x)
    print(' X coordinates of pixel whose value is %d : '%(i), y)


# Also. we can use text file. 
with open('/content/drive/MyDrive/coding_lecture/05.19.CodingLecture7/PennFudanPed/Annotation/FudanPed00001.txt','r') as f:
    txt_data = f.read().splitlines() # split lines by "\n"


for line in txt_data:
    print(line)


import os
import numpy as np
import torch
from PIL import Image

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))



    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:] 

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None] 

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])   # Get coordinates 
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8) # 

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks # For Segmentation task 
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model  = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model


num_classes = 2 # (person + background) 

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

print(model)


