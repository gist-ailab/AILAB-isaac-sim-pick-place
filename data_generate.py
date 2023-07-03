from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.prims import delete_prim
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.objects import DynamicCuboid
import numpy as np
import argparse
from PIL import Image as im
import torchvision.transforms as T
import random


##########
#Read robot name and path#
#########
parser = argparse.ArgumentParser()

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


# camera initialize
hand_camera = Camera(
    prim_path="/World/RGB",
    frequency=20,
    resolution=(1920, 1080),
    position=[0.48176, 0.13541, 0.71],
    orientation=[0.5,-0.5,0.5,0.5]
    # orientation=[1,0,0,0]
)

hand_camera.set_focal_length(1.93)
hand_camera.set_focus_distance(4)

hand_camera.set_horizontal_aperture(2.65)
hand_camera.set_vertical_aperture(1.48)

hand_camera.set_clipping_range(0.01, 10000)
my_world.reset()
hand_camera.initialize()

i = 0
hand_camera.add_distance_to_camera_to_frame()
hand_camera.add_instance_segmentation_to_frame()

stage = get_current_stage()

my_world.reset()

transform = T.ToPILImage()

while simulation_app.is_running():
    my_world.step(render=True)
    # # random 1 ~ 3 data generation in camera boundary
    # if i % 15 == 1:
    #     obj_num = random.randint(1,3)
    #     for l in range(obj_num):
    #         pos_x = (random.random()*0.92+0.02)
    #         pos_y = (random.random()*0.52-0.12)
    #         pos_z = 0.02
    #         cube = DynamicCuboid("/World/object"+str(l),position=[pos_x,pos_y,pos_z/2], scale=[0.02,0.02,0.02])
    #         object_prim=stage.DefinePrim("/World/object"+str(l))
    #         # update semantic information with label 0 is unlabel 1 is background label go for 2 ~
            
    #         add_update_semantics(prim=object_prim, semantic_label=str(l+2))
    # my_world.reset()
        
    # if i % 15 == 12:
    #     hand_rgb_image = hand_camera.get_rgba()[:, :, :3]
    #     hand_depth_image = hand_camera.get_current_frame()["distance_to_camera"]
    #     hand_instance_segmentation_image = hand_camera.get_current_frame()["instance_segmentation"]["data"]
    #     hand_instance_segmentation_dict = hand_camera.get_current_frame()["instance_segmentation"]["info"]["idToSemantics"]
    #     focus_distance = hand_camera.get_focus_distance()
    #     horizontal_aperture = hand_camera.get_horizontal_aperture()
        
    #     print(hand_camera.get_current_frame()["instance_segmentation"])
        
    #     hand_imgplot = transform(hand_rgb_image)
       
    #     # class가 2,3,4로 순서대로 나타나는게 아니라 (2,3) (3,4) 등으로 나타날 때도 있음 해당 예외 처리를 위해 다음과 같은 dict 생성 
    #     class_list = {}
    #     for kl in range(2,5):
    #         if str(kl) in hand_instance_segmentation_dict.keys():
    #             class_list[kl]=int(hand_instance_segmentation_dict[str(kl)]['class'])

    #     # hand_instance_segmentation_image의 경우 class(2,3,4)로 라벨이 되어있음. 이를 label로 바꿔줌
    #     for c in class_list.keys():
    #         np.place(hand_instance_segmentation_image, hand_instance_segmentation_image==c, class_list[c])
    #     print(np.unique(hand_instance_segmentation_image))
    #     # png형태로 저장
    #     hand_inssegplot = im.fromarray(hand_instance_segmentation_image)
    #     if i < 1065:
    #         hand_imgplot.save(args.save_path+"/train/img/img"+str(int(i/15))+".png")
    #         hand_inssegplot.save(args.save_path+"/train/mask/mask"+str(int(i/15))+".png")
    #     else:
    #         hand_imgplot.save(args.save_path+"/val/img/img"+str(int(i/15))+".png")
    #         hand_inssegplot.save(args.save_path+"/val/mask/mask"+str(int(i/15))+".png")
            
    #     for l in range(obj_num):
    #         delete_prim("/World/object"+str(l))
    #     my_world.reset()
        
    # if i == 1500:
    #     simulation_app.close()

    # i += 1
    