# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-4.0 Build COCO like Object Detection dataset 
# ---- ---- ---- ----


import torch
from torch.utils.data import Dataset

import numpy as np

from pycocotools.coco import COCO


class DetectionDataset(Dataset): 
    # def __init__(self, path, img_path):
    def __init__(self, path, transforms=None):
        self.coco = COCO(path)
        # self.img_path = img_path
        self.transforms = transforms
        
        self.image_ids = list(self.coco.imgToAnns.keys())
    
    def __len__(self): 
        return len(self.image_ids)  

    def __getitem__(self, idx): 
        image_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(image_id)[0]['file_name'] # call file's name by using image_id 
        # file_name = os.path.join(self.img_path, file_name)
        # image = Image.open(file_name).convert('RGB')


        annot_ids = self.coco.getAnnIds(imgIds=image_id) # call annotation ids by using image_id 
        annots = [x for x in self.coco.loadAnns(annot_ids) if x['image_id'] == image_id] # call annotations by using annotation id 

        # convert bounding box (x, y, w, h) -> (x1, y1, x2, y2)
        boxes = np.array([annot['bbox'] for annot in annots], dtype=np.float32) 
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = np.array([annot['category_id'] for annot in annots], dtype=np.int32) # class ID of the object
        # masks = np.array([self.coco.annToMask(annot) for annot in annots], dtype=np.uint8) # For segmentation task 

        area = np.array([annot['area'] for annot in annots], dtype=np.float32) # number of pixels inside segmentation mask 
        iscrowd = np.array([annot['iscrowd'] for annot in annots], dtype=np.uint8) #  whether the annotation is for the single object (0) or for the multiple objects

        target = {
            'image_id': image_id,
            'boxes': boxes,
            # 'masks': masks,# For segmentation task 
            'labels': labels,
            'area': area,
            'iscrowd': iscrowd,
        }
    
        # convert numpy array to torch tensor and define data type of tensor
        target["image_id"] = torch.as_tensor(target['image_id'], dtype=torch.uint8)
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        # target['masks'] = torch.as_tensor(target['masks'], dtype=torch.uint8) # For segmentation task 
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target['area'] = torch.as_tensor(target['area'], dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(target['iscrowd'], dtype=torch.uint8)   

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # return image, target 
        return target


dataset = DetectionDataset(path = '//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/instances_val2014.json')


dataset[0]


print('The length of "area" tensor : ')
print(dataset[0]['area'].shape) 

print('The shape of "boxes" tensor : ')
print(dataset[0]['boxes'].shape)


print('Third element of "dataset":' )
print(dataset[3])

print('The length of "area" tensor : ')
print(len(dataset[3]['area']))

print(' "boxes" tensor : ')
print(dataset[3]['boxes'])

print('Data type of "labels" tensor : ')
print(type(dataset[3]['labels']))




print()





