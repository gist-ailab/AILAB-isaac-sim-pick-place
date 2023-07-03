# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-4.0 Finetune Object Detection Model
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





import utils 
from engine import *
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import transforms as T 


def get_transform(train):
    # from torchvision.transforms import transforms as Tr
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


dataset = PennFudanDataset('//content/drive/MyDrive/coding_lecture/05.19.CodingLecture7/PennFudanPed', get_transform(tin=True))
dataset_test = PennFudanDataset('//content/drive/MyDrive/coding_lecture/05.19.CodingLecture7/PennFudanPed', get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist() # Returns a random permutation of integers from 0 to len(dataset) 
print(indices)
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])


batch_size = 2
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)


for i, (img, target) in enumerate(data_loader): 
    print(target)
    break


batch_size = 2
def collate_fn(batch):
    return tuple(zip(*batch))
# used for padding variable-length batches.
# "*" = splat operator : unpacking a list into arguments


res = [['a', 'b', 'c'], [1, 2, 3]]
a = zip(*res)
print(tuple(a))


data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn = collate_fn) #drop_last=False 
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn= collate_fn)


for i, (img, target) in enumerate(data_loader): 
    for j in range(batch_size):
        print('SIze of image : (C, H, W) ',img[j].shape)
        print('Number of the objects :', len(target[j]['boxes']))
    if i> 3: 
        break


print('Number of samples in train dataset :', len(dataset))
print('Number of samples in test dataset :', len(dataset_test))


print('Shape of image tensor :', dataset[0][0].shape) 
print('Bounding box of the objects :', dataset[0][1]['boxes'])
print('Class of the objects :', dataset[0][1]['labels'])
## 1 for person, 2 ,,,,,


print('Number of batches in train dataset :', len(data_loader))
print('Number of batches in test dataset :', len(data_loader_test))


device = "cuda"
model = model.to(device)

params = [p for p in model.parameters() if p.requires_grad] # call model's parameters that require update of gradient

optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1) 

num_epochs = 5


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    # Record Learning rate 
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")) 
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Record Losses 
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # Sum each values of losses
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()











