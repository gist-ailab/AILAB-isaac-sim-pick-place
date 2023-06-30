# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 2-4.0 Object Detection Inference
# ---- ---- ---- ----


import sys
sys.path.append("/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/AILAB-isaac-sim-pick-place/lecture")  
#TODO : change dir
import glob
import random
import argparse

import torch
from PIL import Image, ImageDraw
from detection import YCBDataset, get_model_instance_segmentation, get_transform


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/pth",
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


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 90
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path+"/99.pth"))
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
    print(objects)
    print((prediction[0]['labels']))
    print((prediction[0]))
    for i in range(len(list(prediction[0]['boxes']))):
        if prediction[0]['scores'][i]>0.9:
            print(prediction[0]['boxes'][i])
            draw.multiline_text((list(prediction[0]['boxes'][i])), text = objects[(prediction[0]['labels'][i]-2)].split("/")[-2])
            draw.rectangle((list(prediction[0]['boxes'][i])), outline=(1,0,0),width=3)
    img1.save("bbox.jpg")
    labels = prediction[0]['labels']
    
    for i in labels:
        print(objects[i-2].split("/")[-2])
    # print(objects)
    print("That's it!")


if __name__ == "__main__":
    main()
