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

import os
import sys
import numpy as np
import torch
from PIL import Image
sys.path.append("/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/AILAB-isaac-sim-pick-place/lecture")
from coco.engine import train_one_epoch, evaluate
import coco.utils as utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import coco.transforms as T
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "--img_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/test_img/",
    help="data usd directory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/test_pth/",
    help="img save path directory",
)

args = parser.parse_args()

class YCBDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
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
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
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

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    # our dataset has two classes only - background and person
    num_classes = 90
    # use our dataset and defined transformations
    dataset = YCBDataset(args.img_path+"train/", get_transform(train=True))
    dataset_test = YCBDataset(args.img_path+"val/", get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        if epoch %5 == 4:
            torch.save(model.state_dict(), args.save_path+str(epoch)+".pth")
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        
if __name__ == "__main__":
    main()