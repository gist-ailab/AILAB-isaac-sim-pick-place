# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.isaac.core.prims.xform_prim import XFormPrim
import omni.kit.commands
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import sys
sys.path.append("/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/AILAB-isaac-sim-pick-place/lecture/4-1")
from reach_target_controller import ReachTargetController
from pick_place_controller import PickPlaceController

# from detection.inference_detection import inference_detection
from omni.isaac.examples.ailab_script import AILabExtension
from omni.isaac.examples.ailab_examples import AILab

from detection import YCBDataset, get_model_instance_segmentation, get_transform
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from correct_radial_distortion import depth_image_from_distance_image
# from ggcnn.inferece_ggcnn import inference_ggcnn

import carb
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import glob
import random
import torch
from pathlib import Path


from utils.tasks.pick_place_vision_task import UR5ePickPlace
import coco.transforms as T
import torchvision.transforms as trans

parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/AILAB-isaac-sim-pick-place/lecture/utils/tasks/ur5e_handeye_gripper.usd",
    help="robot usd path.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/AILAB-isaac-sim-pick-place/lecture/dataset/ycb",
    help="data usd directory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/detect_img",
    help="img save path directory",
)

args = parser.parse_args()

my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace(objects_list = [])  # releasing offset at the target position

my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
my_ur5 = my_world.scene.get_object(task_params["robot_name"]["value"])

hand_camera = my_task.get_camera()

stage = get_current_stage()

i = 0
objects = glob.glob(args.data_path+"/*/*.usd")

transform = trans.ToPILImage()

while simulation_app.is_running():
    my_world.step(render=True)
    # random 1 ~ 3 data generation in camera boundary
    if i % 15 == 1:
        obj_num = random.randint(1,3)
        for l in range(obj_num):
            pos_y = (random.random()*0.6-0.17)
            pos_x = (random.random()*0.3+0.33)
            pos_z = 0.00
            a = random.randint(0, len(objects)-1)
            object_prim = create_prim(usd_path=objects[a], prim_path="/World/object"+str(l), position=[pos_x,pos_y,pos_z], scale=[0.2,0.2,0.2])
            # update semantic information with label 0 is unlabel 1 is background label go for 2 ~
            add_update_semantics(prim=object_prim, semantic_label=str(l*100+a+2))
        my_world.reset()
        
    if i % 15== 10:
        
        hand_rgb_image = hand_camera.get_rgba()[:, :, :3]
        hand_depth_image = hand_camera.get_current_frame()["distance_to_camera"]
        hand_instance_segmentation_image = hand_camera.get_current_frame()["instance_segmentation"]["data"]
        hand_instance_segmentation_dict = hand_camera.get_current_frame()["instance_segmentation"]["info"]["idToSemantics"]
        focus_distance = hand_camera.get_focus_distance()
        horizontal_aperture = hand_camera.get_horizontal_aperture()
        
        print(hand_camera.get_current_frame()["instance_segmentation"])
        
        
        hand_imgplot = transform(hand_rgb_image)
       
        # class가 2,3,4로 순서대로 나타나는게 아니라 (2,3) (3,4) 등으로 나타날 때도 있음 해당 예외 처리를 위해 다음과 같은 dict 생성 
        class_list = {}
        for kl in range(2,5):
            if str(kl) in hand_instance_segmentation_dict.keys():
                class_list[kl]=int(hand_instance_segmentation_dict[str(kl)]['class'])
        
        # hand_instance_segmentation_image의 경우 class(2,3,4)로 라벨이 되어있음. 이를 label로 바꿔줌
        for c in class_list.keys():
            np.place(hand_instance_segmentation_image, hand_instance_segmentation_image==c, class_list[c])
        print(np.unique(hand_instance_segmentation_image))
        # png형태로 저장
        hand_inssegplot = Image.fromarray(hand_instance_segmentation_image)
        if i < 100:
            hand_imgplot.save(args.save_path+"/train/img/img"+str(int(i/15))+".png")
            hand_inssegplot.save(args.save_path+"/train/mask/mask"+str(int(i/15))+".png")
        else:
            hand_imgplot.save(args.save_path+"/val/img/img"+str(int(i/15))+".png")
            hand_inssegplot.save(args.save_path+"/val/mask/mask"+str(int(i/15))+".png")
            
        for l in range(obj_num):
            delete_prim("/World/object"+str(l))
        
        
    if i == 150:
        simulation_app.close()

    i += 1
    