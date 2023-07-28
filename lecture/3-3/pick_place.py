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

from omni.isaac.core import World
from omni.kit.viewport.utility import get_active_viewport


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.tasks.pick_place_task import UR5ePickPlace
from utils.controllers.pick_place_controller_robotiq import PickPlaceController

import numpy as np
import random

# YCB Dataset 물체들에 대한 정보 취득
working_dir = os.path.dirname(os.path.realpath(__file__))
ycb_path = os.path.join(Path(working_dir).parent, 'dataset/ycb')
obj_dirs = [os.path.join(ycb_path, obj_name) for obj_name in os.listdir(ycb_path)]
obj_dirs.sort()
object_info = {}
label2name = {}
total_object_num = len(obj_dirs)
for obj_idx, obj_dir in enumerate(obj_dirs):
    usd_file = os.path.join(obj_dir, 'final.usd')
    object_info[obj_idx] = {
        'name': os.path.basename(obj_dir),
        'usd_file': usd_file,
        'label': obj_idx, 
    }
    label2name[obj_idx]=os.path.basename(obj_dir)

# 랜덤한 물체에 대한 usd file path 선택
obje_info = random.sample(list(object_info.values()), 1)
objects_usd = obje_info[0]['usd_file']

# Random하게 생성된 물체들의 ​번호와 카테고리 출력 
print("object: {}".format(obje_info[0]['name']))

# 물체를 생성할 위치 지정(너무 멀어지는 경우 로봇이 닿지 않을 수 있음, 물체 사이의 거리가 가까울 경우 충돌이 발생할 수 있음)
objects_position = np.array([[0.5, 0, 0.1]])
offset = np.array([0, 0, 0.1])

# 물체를 놓을 위치(place position) 지정
target_position = np.array([0.4, -0.33, 0.55])
target_orientation = np.array([0, 0, 0, 1])

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성
my_task = UR5ePickPlace(objects_list = [objects_usd],
                        objects_position = objects_position,
                        offset=offset)

# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

# Task로부터 ur5e 획득
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])

# PickPlace controller 생성
my_controller = PickPlaceController(
    name="pick_place_controller", 
    gripper=my_ur5e.gripper, 
    robot_articulation=my_ur5e
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# 생성한 world 에서 physics simulation step​
while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.is_playing():
        
        # step이 0일때, world와 controller를 reset
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
            
        # my_world로 부터 observation 값들 획득​
        observations = my_world.get_observations()
        
        # 획득한 observation을 pick place controller에 전달
        actions = my_controller.forward(
            picking_position=observations[task_params["task_object_name_0"]["value"]]["position"],
            placing_position=observations[task_params["task_object_name_0"]["value"]]["target_position"],
            current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            end_effector_offset=np.array([0, 0, 0.14])
        )
        
        # controller의 동작이 끝났음을 출력
        if my_controller.is_done():
            print("done picking and placing")
            
        # 선언한 action을 입력받아 articulation_controller를 통해 action 수행. 
        # Controller 내부에서 계산된 joint position값을 통해 action을 수행함​
        articulation_controller.apply_action(actions)

# simulation 종료​
simulation_app.close()
