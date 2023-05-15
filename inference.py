import os
import numpy as np
import torch
from PIL import Image, ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import coco.transforms as T
import glob
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/test_pth",
    help="data usd directory",
)
parser.add_argument(
    "--img_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/test_img/",
    help="img save path directory",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/ycb",
    help="data usd directory",
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 90
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path+"/9.pth"))
    model.eval()
    
    dataset_test = YCBDataset(args.img_path+'val', get_transform(train=False))
    a = random.randint(0,len(dataset_test)-1)
    
    img, _ = dataset_test[a]
    
    with torch.no_grad():
        prediction = model([img.to(device)])
    # save_origin image
    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    img1.save("img.jpg")
    # draw bbox
    draw = ImageDraw.Draw(img1)
    objects = glob.glob(args.data_path+"/*/*.usd")
    print((prediction[0]['labels']))
    for i in range(len(list(prediction[0]['boxes'][:3]))):
        print(prediction[0]['boxes'][i])
        draw.multiline_text((list(prediction[0]['boxes'][i])), text = objects[(prediction[0]['labels'][i]-2)].split("/")[-2])
        draw.rectangle((list(prediction[0]['boxes'][i])), outline=(1,0,0),width=3)
    img1.save("bbox.jpg")
    labels = prediction[0]['labels']
    
    for i in labels:
        print(objects[i-2].split("/")[-2])
    # print(objects)
    print("That's it!")
    
main()
