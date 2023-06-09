from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.utils.semantics import add_update_semantics

import numpy as np
import argparse
from PIL import Image as im
import torchvision.transforms as T
import os
import random
import glob


##########
#Read robot name and path#
#########
parser = argparse.ArgumentParser()
parser.add_argument(
    "--robot_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/github_my/ur5e_handeye_gripper_v2.usd",
    help="robot usd path.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/ycb",
    help="data usd directory",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/test_img",
    help="img save path directory",
)

args = parser.parse_args()


########
#World genration
########
my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()


ur5e_usd_path = args.robot_path

if os.path.isfile(ur5e_usd_path):
    pass
else:
    raise Exception(f"{ur5e_usd_path} not found")

# robot add
add_reference_to_stage(usd_path=ur5e_usd_path, prim_path="/World/UR5")

# camera initialize
hand_camera = Camera(
    prim_path="/World/UR5/realsense/RGB",
    frequency=20,
    resolution=(1920, 1080),
)

my_world.reset()
hand_camera.initialize()

i = 0
hand_camera.add_distance_to_camera_to_frame()
hand_camera.add_instance_segmentation_to_frame()

stage = get_current_stage()

my_world.reset()

transform = T.ToPILImage()

# objects ycd_file list
objects = glob.glob(args.data_path+"/*/*.usd")

while simulation_app.is_running():
    my_world.step(render=True)
    # random 1 ~ 3 data generation in camera boundary
    if i % 150 == 10:
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
        
    if i % 150 == 120:
        
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
        hand_inssegplot = im.fromarray(hand_instance_segmentation_image)
        if i < 10650:
            hand_imgplot.save(args.save_path+"/train/img/img"+str(int(i/15))+".png")
            hand_inssegplot.save(args.save_path+"/train/mask/mask"+str(int(i/15))+".png")
        else:
            hand_imgplot.save(args.save_path+"/val/img/img"+str(int(i/15))+".png")
            hand_inssegplot.save(args.save_path+"/val/mask/mask"+str(int(i/15))+".png")
            
        for l in range(obj_num):
            delete_prim("/World/object"+str(l))
        my_world.reset()
        
    if i == 15000:
        simulation_app.close()

    i += 1
    