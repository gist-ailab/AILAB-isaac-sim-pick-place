# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-4.3 Inference Trained Detection
# reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# ---- ---- ---- ----

#-----0. preliminary -----#

# python path setting
import os
import sys
lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
sys.path.append(lecture_path)

# import packages
from PIL import Image, ImageDraw
import numpy as np
import glob

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import coco.transforms as T
import coco.utils as utils


import random


# get object list
ycb_path = os.path.join(lecture_path, 'dataset/ycb')
obj_dirs = [os.path.join(ycb_path, obj_name) for obj_name in os.listdir(ycb_path)]
obj_dirs.sort()
object_info = {}
total_object_num = len(obj_dirs)
for obj_idx, obj_dir in enumerate(obj_dirs):
    usd_file = os.path.join(obj_dir, 'final.usd')
    object_info[obj_idx] = {
        'name': os.path.basename(obj_dir),
        'usd_file': usd_file,
        'label': obj_idx+2, # set object label 2 ~ 
    }
label2name = {object_info[obj_idx]['label']: object_info[obj_idx]['name'] for obj_idx in object_info.keys()}

#-----1. dataset -----#

# function for getting transform
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# custom dataset class
class YCBDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # convert mask to numpy array
        mask = np.array(mask)
        
        # get unique object ids from mask
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] # remove background
        masks = mask == obj_ids[:, None, None] # split the color-encoded mask into a set of binary masks
        num_objs = len(obj_ids) # number of objects
        
        # get bounding box coordinates for each mask
        boxes = []
        labels = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # 가끔씩 bbox가 같은 좌표로 있는 데이터가 있고 이 같은 경우 학습시 오류 발생, 예외 처리
            if xmin!=xmax and ymin!=ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(obj_ids[i]%100)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # generate labels for coco format
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # suppose all instances are not crowd
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)

#-----2. model -----#
# function for getting model
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

#-----3. inference -----#
if __name__ == "__main__":
    # set paths
    data_root = os.path.join(lecture_path, 'dataset/detect_img')
    test_ckp_path = os.path.join(lecture_path, 'result/ckp')

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load trained model
    num_classes = 90
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    ckp_path = os.path.join(test_ckp_path, 'model_10.pth')
    model.load_state_dict(torch.load())
    model.eval()
    # load test dataset
    test_dataset = YCBDataset(
        root=os.path.join(data_root, 'test'),
        transforms=get_transform(train=False))
    print("Test dataset: ", len(test_dataset))
    
    # get random sample from test dataset
    random_idx = random.randint(0,len(test_dataset)-1)
    img, trg = test_dataset[random_idx]

    # inference
    with torch.no_grad():
        prediction = model([img.to(device)])
    
    # save inference result
    org_img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    org_img.save("original_image.jpg")
    
    # draw bbox
    draw = ImageDraw.Draw(org_img)
    # objects = glob.glob(ycb_path + "/*/*.usd")
    # print(objects)
    print((prediction[0]['labels']))
    print((prediction[0]))
    for i in range(len(list(prediction[0]['boxes']))):
        if prediction[0]['scores'][i]>0.9:
            print(prediction[0]['boxes'][i])
            predict_label = prediction[0]['labels'][i]
            draw.multiline_text((list(prediction[0]['boxes'][i])), text = label2name[predict_label])
            draw.rectangle((list(prediction[0]['boxes'][i])), outline=(1,0,0),width=3)
    org_img.save("visualize_bbox.jpg")
    
    labels = prediction[0]['labels']
    for i in labels:
        print(label2name[i])
    # print(objects)
    print("That's it!")
