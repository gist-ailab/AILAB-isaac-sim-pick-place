# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.7 Basic simulation loop with camera (RGBD/Mask)
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera                            #
from omni.isaac.core.utils.semantics import add_update_semantics

import os                                                       #
import sys
from PIL import Image  
import numpy as np                                      #
import random
import copy
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)
print(lecture_path)

my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

for i in range(3):
    scale = list(np.random.rand(3) * 0.2)               
    position = [random.random()*0.3+0.33, random.random()*0.6-0.17, 0.1]                      
    cube = DynamicCuboid(
        prim_path="/World/object"+str(i), 
        position=position, 
        scale=scale)
    add_update_semantics(prim=cube.prim, semantic_label=str(i+2))
my_world.reset()

save_root = os.path.join(lecture_path, "2-2/sample_data")            
print("Save root: ", save_root)                                 
os.makedirs(save_root, exist_ok=True)                           

def save_image(image, path, mode = None):                                    #
    image = Image.fromarray(image, mode)                              
    image.save(path)                                            

my_camera = Camera(                                             
    prim_path="/World/RGB",                                     
    frequency=20,                                               
    resolution=(1920, 1080),                                    
    position=[0.48176, 0.13541, 0.71],  
    orientation=[0.5,-0.5,0.5,0.5]                
)                                                               
my_camera.set_focal_length(1.93)                                
my_camera.set_focus_distance(4)                                 
my_camera.set_horizontal_aperture(2.65)                         
my_camera.set_vertical_aperture(1.48)                           
my_camera.set_clipping_range(0.01, 10000)                      
my_camera.add_instance_segmentation_to_frame()
my_camera.initialize()                                          

ep_num = 0
max_ep_num = 10


while simulation_app.is_running():
    
    ep_num += 1                                      #
    my_world.step(render=True)
    print("Episode: ", ep_num)

    rgb_image = my_camera.get_rgba()                        #
    instance_segmentation_image = my_camera.get_current_frame()["instance_segmentation"]["data"]
    instance_segmentation_dict = my_camera.get_current_frame()["instance_segmentation"]["info"]["idToSemantics"]

    if ep_num == max_ep_num:
        print(my_camera.get_current_frame()["instance_segmentation"])
        save_image(rgb_image, os.path.join(save_root, "semantic_img.png"))  #
        
        origin_img_r = copy.deepcopy(instance_segmentation_image)
        origin_img_g = copy.deepcopy(instance_segmentation_image)
        origin_img_b = copy.deepcopy(instance_segmentation_image)
        
        
        np.place(origin_img_r, origin_img_r==2, 255)
        np.place(origin_img_g, origin_img_g==3, 255)
        np.place(origin_img_b, origin_img_b==4, 255)
        
        np.place(origin_img_r, origin_img_r!=255, 0)
        np.place(origin_img_g, origin_img_g!=255, 0)
        np.place(origin_img_b, origin_img_b!=255, 0)
        
        origin_img = np.array([origin_img_r,origin_img_g,origin_img_b])
        print(rgb_image.shape,origin_img.shape)
        
        save_image(origin_img.astype(np.uint8).transpose(1,2,0), os.path.join(save_root, "visualize_semantic_mask.png"))
        simulation_app.close()
simulation_app.close()