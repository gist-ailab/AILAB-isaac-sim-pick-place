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

from utils.tasks.pick_place_task import UR5ePickPlace
from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_camera_path
import numpy as np
import glob, random

# get ycb directories to sys.path
working_dir = os.path.dirname(os.path.realpath(__file__))
objects_path = os.path.join(Path(working_dir).parent, "dataset/ycb/*/*.usd")
objects_list = glob.glob(objects_path)
objects_list = random.sample(objects_list, 3)

# if you don't declare objects_position, the objects will be placed randomly
objects_position = np.array([[0.3, 0.3, 0.1],
                             [-0.3, 0.3, 0.1],
                             [-0.3, -0.3, 0.1]])
offset = np.array([0, 0, 0.1])  # releasing offset at the target position
target_position = np.array([0.4, -0.33, 0.55])  # 0.55 for considering the length of the gripper tip
target_orientation = np.array([0, 0, 0, 1])

my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace(objects_list = objects_list,
                        objects_position = objects_position,
                        offset=offset)  # releasing offset at the target position
# my_task = UR5ePickPlace(objects_position = objects_position)
# my_task = UR5ePickPlace()
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_ur5 = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_ur5.gripper, robot_articulation=my_ur5
    )
articulation_controller = my_ur5.get_articulation_controller()

viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=observations[task_params["task_object_name_0"]["value"]]["position"],
            placing_position=observations[task_params["task_object_name_0"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0, 0.15])
        )
        # print('---------------------------------------------------------')
        # print(f'EE position: {observations["my_ur5e"]["end_effector_position"][:2]}')
        # print(f'cube position: {observations["cube"]["position"][:2]}')
        # print(f'picking position: {observations[task_params["task_object_name_0"]["value"]]["position"][:2]}')
        # print('---------------------------------------------------------\n\n')
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
simulation_app.close()
