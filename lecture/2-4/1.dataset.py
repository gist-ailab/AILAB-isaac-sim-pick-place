# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-4.1 Custom YCB Dataset for detection
# ---- ---- ---- ----

#-----0. preliminary -----#

# python path setting
import os
import sys
lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
sys.path.append(lecture_path)

# import packages
import torch
from PIL import Image
import numpy as np
import coco.transforms as T

# function for getting transform
def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


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


if __name__=="__main__":
    lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
    data_root = os.path.join(lecture_path, 'dataset/detect_img')

    train_dataset = YCBDataset(
        root=os.path.join(data_root, 'train'),
        transforms=get_transform(train=True))
    print("Train dataset: ", len(train_dataset))

    test_dataset = YCBDataset(
        root=os.path.join(data_root, 'test'),
        transforms=get_transform(train=False))
    print("Test dataset: ", len(test_dataset))


