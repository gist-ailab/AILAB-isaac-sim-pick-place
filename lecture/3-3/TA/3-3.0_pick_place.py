# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day3. 
# 3-3.0 pick and place with YCB object to random pose
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})


import sys, os
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
directory = Path(current_dir).parent
sys.path.append(str(directory))


from utils.tasks.pick_place_task import UR5ePickPlace
from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from omni.isaac.core import World

from omni.kit.viewport.utility import get_active_viewport
import numpy as np
import glob, random                                                                 #


working_dir = os.path.dirname(os.path.realpath(__file__))                           #
objects_path = os.path.join(Path(working_dir).parent, "dataset/ycb/*/*.usd")        #
objects_list = glob.glob(objects_path)                                              #
objects_list = random.sample(objects_list, 3)                                       #


objects_position = np.array([[0.3, 0.3, 0.1],
                             [-0.3, 0.3, 0.1],
                             [-0.3, -0.3, 0.1]])
target_position = np.array([0.4, -0.33, 0.55])
target_orientation = np.array([0, 0, 0, 1])
offset = np.array([0, 0, 0.1])

my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace(objects_list = objects_list,
                        objects_position = objects_position,
                        offset=offset)

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
            end_effector_offset=np.array([0.125, 0.095, 0.03]),
        )
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
simulation_app.close()
