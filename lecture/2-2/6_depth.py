# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

# add necessary directories to sys.path
import sys, os
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
directory = Path(current_dir).parent
sys.path.append(str(directory))

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_camera_path
import numpy as np
from PIL import Image
# import matplotlib
# # matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

scale = list(np.random.rand(3) * 0.2)               
position = [0.3, 0.3, scale[2]/2]                       
cube = DynamicCuboid(
    prim_path="/World/object", 
    position=position, 
    scale=scale)

my_world.reset()

save_root = os.path.join(os.getcwd(), "sample_data")  
print("Save root: ", save_root)
os.makedirs(save_root, exist_ok=True)

def save_image(image, path):                                    #
    image = Image.fromarray(image)                              #
    image.save(path)                                            #

my_camera = Camera(                                             #
    prim_path="/World/RGB",                                     #
    frequency=20,                                               #
    resolution=(1920, 1080),                                    #
    position=[0.48176, 0.13541, 0.71], # attach to robot hand   #
    orientation=[0.5,-0.5,0.5,0.5] # quaternion                 #
)                                                               #
my_camera.set_focal_length(1.93)                                #
my_camera.set_focus_distance(4)                                 #
my_camera.set_horizontal_aperture(2.65)                         #
my_camera.set_vertical_aperture(1.48)                           #
my_camera.set_clipping_range(0.01, 10000)                       #
my_camera.add_distance_to_camera_to_frame()

my_camera.initialize()        


# viewport = get_active_viewport()
# viewport.set_active_camera('/World/ur5e/realsense/Depth')
# viewport.set_active_camera('/OmniverseKit_Persp')

i = 0


ep_num = 0
max_ep_num = 100

while simulation_app.is_running():    
    ep_num += 1                                      #
    my_world.step(render=True)
    print("Episode: ", ep_num)

    rgb_image = my_camera.get_rgba()
    depth_image = my_camera.get_current_frame()["distance_to_camera"]

    if ep_num == max_ep_num:
        depth_image[depth_image < 0.67] = 0.67
        depth_image = depth_image - 0.5
        depth_image = (depth_image*255*2).astype(np.uint8)
        save_image(rgb_image, os.path.join(save_root, "rgb{}.png".format(ep_num)))
        save_image(depth_image, os.path.join(save_root, "depth{}.png".format(ep_num)))

        simulation_app.close()
