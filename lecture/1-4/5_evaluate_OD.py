# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-4.0 Evaluate Object Detection Model
# ---- ---- ---- ----


from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import utils 
from engine import *

def evaluate(model, data_loader, device):
    # FIXME remove this and make paste_masks_in_image run on the GPU
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator




evaluate(model, data_loader_test, device=device)


import cv2
def overlay_instances(img, prediction, threshold=0.8):
    label_dict = {0 : 'background', 1: 'person'}
    ori_img = img.mul(255).permute(1,2,0).byte().numpy() # Tensor * 255, Convert RGB -> BGR, Tensor to numpy
    for idx in range(len(prediction[0]['boxes'])):
        score = prediction[0]['scores'][idx].cpu().detach().numpy().item() # detach() : Generation of tensors that do not propagate gradients from existing sensors
        if score < threshold: 
            continue
        box = x1,y1,x2,y2 = prediction[0]['boxes'][idx].cpu().detach().numpy()
        label = prediction[0]['labels'][idx].cpu().detach().numpy().item()
        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255,0,0),2) 
        cv2.rectangle(ori_img, (int(x1), int(y1)), (int(x1+60), int(y1+20)), (255,0,0),-1)
        cv2.putText(ori_img, label_dict[label], (int(x1), int(y1+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1)
    return Image.fromarray(ori_img)



''' Inference on single image '''
img, _ = dataset_test[0]
model.eval() 
with torch.no_grad(): 
    prediction = model([img.to(device)])
print(prediction)


results = overlay_instances(img, prediction, threshold=0.8)
results




torch.save(model.state_dict(), './person_detector.pth')



ckpts = torch.load('./person_detector.pth')
model.load_state_dict(ckpts) 


