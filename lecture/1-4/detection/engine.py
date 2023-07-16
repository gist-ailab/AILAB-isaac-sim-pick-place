import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

from utils import *
from tqdm import tqdm
from pprint import PrettyPrinter, pprint

from torchmetrics.detection import MeanAveragePrecision

classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
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
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
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

def evaluate(model, test_loader, device):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    model.eval()
    total_count = 0

    with torch.no_grad():
        # Batches
        for i, (images, targets) in enumerate(tqdm(test_loader, desc='Evaluating')):
            if len(images)==1:
                images = images[0].unsqueeze(0)
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [target['boxes'] for target in targets]
            labels = [target['labels'] for target in targets]
            targets = [{'boxes':torch.stack(boxes).squeeze().to(device), 'labels':torch.stack(labels).squeeze().to(device)}]
            
            print(targets[0]['boxes'])
            print(targets[0]['boxes'].shape)
            print(targets[0]['labels'])
            print(targets[0]['labels'].shape)

            predictions = model(images)
            predicted_locs = predictions[0]['boxes']
            predicted_scores = predictions[0]['scores']
            predicted_labels = predictions[0]['labels']
            print(predicted_locs.shape)
            print(predicted_labels.shape)
            print(predicted_scores.shape)
            # predicted_locs, predicted_scores = model(images)

            metric = MeanAveragePrecision(iou_type = 'bbox', iou_thresholds=[0.5, 0.75])
            metric.update(predictions, targets)
            if i == 0:
                map_dict = metric.compute()
            else:
                result = metric.compute()
                map_dict = {key:map_dict[key]+result[key] for key in map_dict.keys()}
                
            total_count += len(targets)
        map_dict = {key:map_dict[key]/total_count for key in map_dict.keys()}
        pprint(map_dict)