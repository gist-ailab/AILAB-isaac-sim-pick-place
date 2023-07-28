from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.kit.viewport.utility import get_active_viewport

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.tasks.pick_place_task import UR5ePickPlace
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController

# if you don't declare objects_position, the objects will be placed randomly
objects_position = np.array([0.4, 0.4, 0.1])
target_position = np.array([0.4, -0.33, 0.05])  # 0.55 for considering the length of the gripper tip
target_orientation = np.array([0, 0, 0, 1])
offset = np.array([0, 0, 0.1])  # releasing offset at the target position

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

# 그리퍼 열기 / 닫기 명령어 입력받음
while True:
    instruction = input('Enter the instruction [open/close]:')
    if instruction in ["o", "open", "c", "close"]:
        break
    else:    
        print("wrong instruction")
print('instruction : ', instruction)

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        observations = my_world.get_observations()
        if instruction == "o" or instruction == "open":
            actions = my_controller.open(
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            )
        elif instruction == "c" or instruction == "close":
            actions = my_controller.close(
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            )

        articulation_controller.apply_action(actions)
        # 컨트롤러가 끝나면 새로운 명령어 입력 받음
        if my_controller.is_done():
            if instruction == "o" or instruction == "open":
                print("done opening the gripper\n")
            elif instruction == "c" or instruction == "close":
                print("done closing the gripper\n")

            while True:
                instruction = input('Enter the instruction [open/close/quit]:')
                if instruction in ["o", "open", "c", "close", "q", "quit"]:
                    break
                else:    
                    print("wrong instruction")
            print('instruction : ', instruction)
            print()
            if instruction == 'q' or instruction == 'quit':
                break
            my_controller.reset()
simulation_app.close()
