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

from utils.tasks.basic_task import SetUpUR5eObject
from omni.isaac.core import World
from utils.controllers.ee import EndEffectorController
from omni.kit.viewport.utility import get_active_viewport

my_world = World(stage_units_in_meters=1.0)
my_task = SetUpUR5eObject()
my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = EndEffectorController(
    name="end_effector_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e
    )

articulation_controller = my_ur5e.get_articulation_controller()
my_controller.reset()

viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

ep_num = 0
max_ep_num = 100

while simulation_app.is_running():
    ep_num += 1                                      #
    my_world.step(render=True)

    if ep_num == max_ep_num:
        break
simulation_app.close()
