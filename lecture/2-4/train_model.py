# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-4.2 Train Detection Model with Custom YCB Dataset
# reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# ---- ---- ---- ----

#-----0. preliminary -----#

# python path setting
import os
import sys
lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
sys.path.append(lecture_path)

# import packages
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from coco.engine import train_one_epoch, evaluate
import coco.utils as utils

from custom_dataset import *

#-----2. model -----#
# function for getting model
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model_object_detection(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


#-----3. training -----#
if __name__=="__main__":
    # set paths
    data_root = os.path.join(lecture_path, 'dataset/detect_img')
    test_ckp_path = os.path.join(lecture_path, 'result/ckp')
    
    if not os.path.isdir(test_ckp_path):
        os.mkdir(test_ckp_path)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Train device: ", device)
    
    # use our dataset and defined transformations
    train_dataset = YCBDataset(
        root=os.path.join(data_root, 'train'),
        transforms=get_transform(train=True))
    print("Train dataset: ", len(train_dataset))
    
    test_dataset = YCBDataset(
        root=os.path.join(data_root, 'test'),
        transforms=get_transform(train=False))
    print("Test dataset: ", len(test_dataset))

    # define training and validation data loaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=8,
        collate_fn=utils.collate_fn)

    # get number of classes
    num_classes = 43

    # get the model using our helper function
    model = get_model_object_detection(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=30,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
        if epoch%5==4:
            torch.save(model.state_dict(), os.path.join(test_ckp_path, 'model_{}.pth'.format(epoch)))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)