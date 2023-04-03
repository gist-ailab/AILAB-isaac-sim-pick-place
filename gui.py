# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# from tkinter import *
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.utils.semantics import add_update_semantics
from pick_place_controller import PickPlaceController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.sensor import Camera
import carb
import sys
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob
import random

# tk = TK()
# tk.mainloop()
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

ur5e_usd_path = "/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/github_my/isaac-sim-pick-place/ur5e_handeye_gripper.usd"
if os.path.isfile(ur5e_usd_path):
    pass
else:
    raise Exception(f"{ur5e_usd_path} not found")

add_reference_to_stage(usd_path=ur5e_usd_path, prim_path="/World/UR5e")
gripper = ParallelGripper(
    end_effector_prim_path="/World/UR5e/right_inner_finger_pad",
    joint_prim_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0.0, 0.0]),
    joint_closed_positions=np.array([np.pi*2/9, -np.pi*2/9]),
    action_deltas=np.array([-np.pi*2/9, np.pi*2/9]),
)
my_ur5e = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/UR5e", name="my_ur5e", end_effector_prim_name="right_inner_finger_pad", gripper=gripper
    )
)

size_z = 0.3

pos0_y = (random.random()*1.2+0.8)/4
pos0_x = (random.random()*1.2-0.4)/4
pos0_z = size_z

pos1_y = (random.random()*1.2+0.8)/4
pos1_x = (random.random()*1.2-0.4)/4
pos1_z = size_z

pos2_y = (random.random()*1.2+0.8)/4
pos2_x = (random.random()*1.2-0.4)/4
pos2_z = size_z

objects = glob.glob("/home/ailab/Workspace/minhwan/ycb/*/*.usd")
# for i in range(3):
a = random.randint(0,len(objects)-1)
cube1 = create_prim(usd_path=objects[a], prim_path="/World/object0", position=[pos0_x,pos0_y,pos0_z], scale=[0.3,0.3,0.3])

b = random.randint(0,len(objects)-1)
cube2 = create_prim(usd_path=objects[b], prim_path="/World/object1", position=[pos1_x,pos1_y,pos1_z], scale=[0.3,0.3,0.3])

c = random.randint(0,len(objects)-1)
cube3 = create_prim(usd_path=objects[c], prim_path="/World/object2", position=[pos2_x,pos2_y,pos2_z], scale=[0.3,0.3,0.3])


stage = get_current_stage()
obj0_prim = stage.DefinePrim("/World/object0")
add_update_semantics(prim=obj0_prim, semantic_label=str(0))

stage = get_current_stage()
obj1_prim = stage.DefinePrim("/World/object1")
add_update_semantics(prim=obj1_prim, semantic_label=str(1))

stage = get_current_stage()
obj2_prim = stage.DefinePrim("/World/object2")
add_update_semantics(prim=obj2_prim, semantic_label=str(2))


my_world.scene.add_default_ground_plane()
my_ur5e.gripper.set_default_state(my_ur5e.gripper.joint_opened_positions)
my_world.reset()

my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e, end_effector_initial_height=0.3
)
articulation_controller = my_ur5e.get_articulation_controller()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        
        if my_controller._event == 0:
            if my_controller._t >= 0.99:
                my_controller.pause()
                
                
                my_controller.resume()
    
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=np.array([pos1_x,pos1_y, 0.00]),
            placing_position=np.array([0.4, -0.33, 0.02]),
            current_joint_positions=my_ur5e.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.25]),
        )
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
    if args.test is True:
        break


simulation_app.close()