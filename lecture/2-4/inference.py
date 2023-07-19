# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-4.3 Inference Trained Detection
# reference: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# ---- ---- ---- ----

#-----0. preliminary -----#

# python path setting
from custom_dataset import *
from train_model import *

lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
sys.path.append(lecture_path)

# import packages
from PIL import Image, ImageDraw

import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

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

#-----3. inference -----#
if __name__ == "__main__":
    # set paths
    data_root = os.path.join(lecture_path, 'dataset/detect_img')
    test_img_path = os.path.join(lecture_path, 'result/img')
    test_ckp_path = os.path.join(lecture_path, 'result/ckp')
    
    if not os.path.isdir(test_img_path):
        os.mkdir(test_img_path)

    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load trained model
    num_classes = 43
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    ckp_path = os.path.join(test_ckp_path, '100.pth')
    model.load_state_dict(torch.load(ckp_path))
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
    org_img.save(os.path.join(test_img_path, "org_img.jpg"))
    
    # draw bbox
    draw = ImageDraw.Draw(org_img)
    # objects = glob.glob(ycb_path + "/*/*.usd")
    # print(objects)
    print((prediction[0]['labels']))
    print((prediction[0]))
    for i in range(len(list(prediction[0]['boxes']))):
        if prediction[0]['scores'][i]>0.5:
            print(prediction[0]['boxes'][i])
            predict_label = prediction[0]['labels'][i]
            draw.multiline_text((list(prediction[0]['boxes'][i])), text = label2name[predict_label])
            draw.rectangle((list(prediction[0]['boxes'][i])), outline=(1,0,0),width=3)
    org_img.save(os.path.join(test_img_path, "visualize_bbox.jpg"))
    
    labels = prediction[0]['labels']
    for i in labels:
        print(label2name[i])
    # print(objects)
    print("That's it!")