# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day1. 
# 1-4.0 Public Object Detection dataset 
# ---- ---- ---- ----


# Let's open "COCO format" annotations with Python
import json
with open('//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/instances_val2014.json', 'r') as f: 
    data = json.load(f) 

print(' Data type of the file :', type(data))
print(' Keys of the dictionary :', data.keys())

print(data['info'])

print(type(data['images']))
print(data['images'][0]) # print first element of "images"
print(type(data['images'][0]['height'])) # print data type of "height" of the first element
print(type(data['images'][0]['id']))


#
first_annotations = data['annotations'][0]
print(data['annotations'][0])

print(first_annotations['image_id'])

print(type(first_annotations['bbox']))

print(type(data['categories']))

print(data['categories'])


#
from pycocotools.coco import COCO

coco = COCO('//content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/instances_val2014.json')

image_ids = list(coco.imgToAnns.keys()) # list of image ids 

idx = 0 
image_id = image_ids[idx]
file_name = coco.loadImgs(image_id)[0]['file_name'] # call file name by using image id 

print('The first image')
print('Image id :', image_id, '  File name : ', file_name)


# Load annotations by using image id 
annot_ids = coco.getAnnIds(imgIds = image_id) 

annots = [x for x in coco.loadAnns(annot_ids) if x['image_id'] == image_id]
print('Annotations :' , annots)

print('Number of objects :' , len(annots))
print('Keys of the annotation :', annots[0].keys())


import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image

def crop_instances(img_path, annotations):
    original_img = np.array(Image.open(img_path))
    plt.title('Original Image') # Assign title
    plt.axis('off') # Turn off the axis
    plt.imshow(original_img)
    plt.figure(figsize=(16,9)) # Assign size of figure
    for idx in range(0, len(annotations)):
        plt.subplot(5,3,idx+1) # assign row, column , index of position
        plt.axis('off') 
        x,y,w,h = bbox = annotations[idx]['bbox'] 
        crop_img = original_img[int(y):int(y+h), int(x):int(x+w), :] # crop area of each object
        plt.imshow(crop_img)


img_path = "/content/drive/MyDrive/coding_lecture/04.14.CodingLecture6/coco_val/COCO_val2014_000000558840.jpg"
crop_instances(img_path, annots)





print()