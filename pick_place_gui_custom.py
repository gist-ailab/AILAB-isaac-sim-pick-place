# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


#
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from tasks.pick_place_task import UR5ePickPlace
from controllers.pick_place_controller_robotiq import PickPlaceController
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_camera_path
import numpy as np
import glob, os, random


# ---- ---- ---- ----      ---- ---- ---- ----


# = gui + 
import os
from omni.isaac.examples.ailab_script import AILabExtension
from omni.isaac.examples.ailab_examples import AILab


class AILabExtensions(AILabExtension):
    def __init__(self):
        super().__init__()

    def on_startup(self, ext_id: str):
        super().on_startup(ext_id)
        super().start_extension(
            menu_name="",
            submenu_name="",
            name="AILab extension",
            title="AILab extension Example",
            doc_link="https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_core_hello_world.html",
            overview="This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            file_path=os.path.abspath(__file__),
            sample=AILab(),
        )
        return

gui_test = AILabExtensions()
gui_test.on_startup(ext_id='omni.isaac.examples-1.5.1')


# ---- ---- ---- ----      ---- ---- ---- ----


from pathlib import Path
CURRENT_FILE = Path(__file__).parent
total_objects_list = glob.glob(f"{CURRENT_FILE}/ycb_usd/*/*/*.usd")     # : !!
current_num_total_objects = len(total_objects_list)
assert current_num_total_objects!=0, f"!!>> place YCB objects in folder! now : {current_num_total_objects}"

# working_dir = os.path.dirname(os.path.realpath(__file__))

# objects_list = glob.glob(objects_path)
objects_list = random.sample(total_objects_list, 3)
# get three objects randomly


#
objects_position = np.array([[ 0.3,  0.3, 0.1],
                             [ 0.3,  0.0, 0.1],
                             [ 0.4,  0.5, 0.1]])
offset = np.array([0, 0, 0.1])
target_position = np.array([0.4, -0.33, 0.55])  # 0.55 for considering the length of the gripper tip
target_orientation = np.array([0, 0, 0, 1])


#
my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace(objects_list = objects_list,
                        objects_position = objects_position,
                        offset=offset)  # releasing offset at the target position


#
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


# ---- ---- ---- ----      ---- ---- ---- ----


#
i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        
        if gui_test.use_custom_updated:
            observations = my_world.get_observations()
            
            #TODO: check gui_test.current_target is exist

            actions = my_controller.forward(
                picking_position=observations[task_params[gui_test.current_target]["value"]]["position"],
                placing_position=observations[task_params[gui_test.current_target]["value"]]["target_position"],
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=np.array([0.125, 0.095, 0.04]),
                # end_effector_offset=np.array([0, 0, 0.04]),
            )
            if my_controller.is_done():
                print("\n done picking and placing \n")
                print("resetting sim \n")
                my_world.reset()
                my_controller.reset()
                gui_test.use_custom_updated = False
                gui_test.current_target = None
            articulation_controller.apply_action(actions)
        else:
            observations = my_world.get_observations()

simulation_app.close()
