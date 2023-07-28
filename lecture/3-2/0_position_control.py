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
from omni.isaac.core import World
from utils.controllers.basic_manipulation_controller import BasicManipulationController
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from omni.kit.viewport.utility import get_active_viewport
import numpy as np


my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace()
my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = BasicManipulationController(
    name='basic_manipulation_controller',
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_ur5e, attach_gripper=True
    ),
    gripper=my_ur5e.gripper,
    events_dt=[0.008],
)

articulation_controller = my_ur5e.get_articulation_controller()
my_controller.reset()

viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

max_step = 150

ee_target_position = np.array([0.25, -0.23, 0.2]) 

while simulation_app.is_running():
    my_world.step(render=True)

    if my_world.is_playing():
        if my_world.current_time_step_index < max_step:
            observations = my_world.get_observations()

            actions = my_controller.forward(
                target_position=ee_target_position,
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset = np.array([0, 0, 0.14])
            )

            if my_controller.is_done():
                print("done position control of end-effector")
                break
            articulation_controller.apply_action(actions)
simulation_app.close()
