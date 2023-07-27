import os
import sys
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)

from PIL import Image
import copy
import numpy as np
from torchvision.transforms import transforms

def save_image(image, path):                                    
    image = Image.fromarray(image)                              
    image.save(path)                                            

mask_array = Image.open(lecture_path+"/2-2/sample_data/mask/mask0.png")

trans = transforms.ToTensor()
mask_array = trans(mask_array)
mask_array = mask_array.numpy()

cls_list = np.unique(mask_array)[1:]

origin_img = np.zeros((3,1080,1920))

for idx, cls in enumerate(cls_list):
    origin_img_r = copy.deepcopy(mask_array)

    np.place(origin_img_r, origin_img_r==cls, 255)
    np.place(origin_img_r, origin_img_r!=255, 0)

    origin_img[idx] = origin_img_r

save_image(origin_img.astype(np.uint8).transpose(1,2,0), os.path.join(lecture_path, "2-2/sample_data/visualize_semantic_mask.png"))