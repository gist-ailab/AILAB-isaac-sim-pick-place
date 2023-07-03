# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.3 Basic simulation loop with primitive object load
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects import DynamicCuboid       #

from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.prims.rigid_prim import RigidPrim

from omni.isaac.core.utils.prims import create_prim, delete_prim
import numpy as np                                      #
import glob
import argparse

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


scene = Scene()
scene.add_default_ground_plane()


max_ep_step = 15                                        #
ep_num = 0                                              #
max_ep_num = 100                                        #

obj_idx = 0                                             #

x = XFormPrim("/World/x", position=[10,0,0])
RigidPrim("/World/x", linear_velocity=[0.01,0,0])
y = XFormPrim("/y", position=[0,10,0])
z = XFormPrim("/z", position=[0,0,10])
while simulation_app.is_running():
    my_world.step(render=True)
    
# objects = glob.glob(args.data_path+"/*/*.usd")
exit()
while simulation_app.is_running():
    
    scale = list(np.random.rand(3) * 0.5)               #
    position = [0, 0, scale[2]/2]                       #
    my_world.step(render=True)
    
    if obj_idx % 60000 == 0:
        cube = DynamicCuboid("/x/object"+str(obj_idx), position=position, scale=scale) 
    if obj_idx % 60000 == 10000:
        delete_prim("/World/x/object"+str(obj_idx-100))
    
    if obj_idx % 60000 == 20000:
        cube = DynamicCuboid("/World/y/object"+str(obj_idx), position=position, scale=scale) 
    if obj_idx % 60000 == 30000:
        delete_prim("/World/y/object"+str(obj_idx-100))
    
    # my_world.step(render=True)
    if obj_idx % 60000 == 40000:
        cube = DynamicCuboid("/World/z/object"+str(obj_idx), position=position, scale=scale) 
    if obj_idx % 60000 == 50000:
        delete_prim("/World/z/object"+str(obj_idx-100))
    
    
    # print("End Episode: ", ep_num)                      #
    
    obj_idx += 1 
    if ep_num >= max_ep_num:                            #
        simulation_app.close()