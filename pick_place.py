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

from tasks.pick_place import PickPlace
from controllers.pick_place_controller_robotiq import PickPlaceController
from omni.isaac.core import World
import numpy as np

my_world = World(stage_units_in_meters=1.0)
my_task = PickPlace()
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_ur5 = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = PickPlaceController(name="pick_place_controller", gripper=my_ur5.gripper, robot_articulation=my_ur5)
articulation_controller = my_ur5.get_articulation_controller()

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=observations[task_params["cube_name"]["value"]]["position"],
            placing_position=observations[task_params["cube_name"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0, 0.02]),
        )
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
simulation_app.close()
