# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.6 Basic simulation loop with camera (RGBD)
# ---- ---- ---- ----

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

# add necessary directories to sys.path
import sys, os
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
import numpy as np
from PIL import Image


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

save_root = os.path.join(lecture_path, "2-2/sample_data")  
print("Save root: ", save_root)
os.makedirs(save_root, exist_ok=True)

def save_image(image, path):                                    
    image = Image.fromarray(image)                              
    image.save(path)                                            


def depth_image_from_distance_image(distance, intrinsics):
    """Computes depth image from distance image.
    
    Background pixels have depth of 0
    
    Args:
        distance: HxW float array (meters)
        intrinsics: 3x3 float array
    
    Returns:
        z: HxW float array (meters)
    
    """
    fx = intrinsics[0][0]
    cx = intrinsics[0][2]
    fy = intrinsics[1][1]
    cy = intrinsics[1][2]
    
    height, width = distance.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    
    x_over_z = (px - cx) / fx
    y_over_z = (py - cy) / fy
    
    z = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    return z


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

camera_intrinsics = my_camera.get_intrinsics_matrix()
print('camera_intrinsics: ', camera_intrinsics)

ep_num = 0
max_ep_num = 12

while simulation_app.is_running():    
    ep_num += 1                                      #
    my_world.step(render=True)
    print("Episode: ", ep_num)

    rgb_image = my_camera.get_rgba()
    distance_image = my_camera.get_current_frame()["distance_to_camera"]

    if ep_num == max_ep_num:
        distance_image = (distance_image*255).astype(np.uint8)
        depth_image = depth_image_from_distance_image(distance_image, camera_intrinsics).astype(np.uint8)
        save_image(rgb_image, os.path.join(save_root, "rgb.png"))
        save_image(distance_image, os.path.join(save_root, "distance.png"))
        save_image(depth_image, os.path.join(save_root, "depth.png"))

        simulation_app.close()
