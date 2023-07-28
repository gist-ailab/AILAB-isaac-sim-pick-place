
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.universal_robots.controllers import RMPFlowController

import os
import sys
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)


import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.controllers.basic_manipulation_controller import BasicManipulationController
from utils.tasks.basic_task import SetUpUR5eObjectCamera


# 경로 셋팅
save_root = os.path.join(lecture_path, "3-2/sample_data")            
print("Save root: ", save_root)                                 
os.makedirs(save_root, exist_ok=True)  

# World 생성
my_world = World(stage_units_in_meters=1.0)


# Task 생성
my_task = SetUpUR5eObjectCamera()

# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

# Task로부터 ur5e와 camera를 획득
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
camera = my_task.get_camera()

# PickPlace controller 생성
my_controller = BasicManipulationController(
    name='basic_manipulation_controller',
    cspace_controller=RMPFlowController(
        name="basic_manipulation_controller_cspace_controller", robot_articulation=my_ur5e, attach_gripper=True
    ),
    gripper=my_ur5e.gripper,
    events_dt=[0.008],
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()


# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# target object를 찾았는지 확인하기 위해서 found_obj 선언(시작시에는 찾지 못한 상태이니 False)
found_obj = False
print('search-target')


ep_num = 0

# theta 값에 따라서 움직이며, target object를 찾을때까지 360도를 회전
# theta 값을 계속해서 달라지게 하기 위해 for문 사용(예제는 45도씩 회전하도록 하였음)
for theta in range(0, 360, 45):
    # theta 값에 따라서 end effector의 위치를 지정(x, y, z)
    r, z = 4, 0.3
    x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
    
    # 생성한 world 에서 physics simulation step​
    while simulation_app.is_running():
        ep_num += 1
        my_world.step(render=True)
        if my_world.is_playing():
            observations = my_world.get_observations()

            actions = my_controller.forward(
                target_position=np.array([x, y, z]),
                    current_joint_positions=my_ur5e.get_joint_positions(),
                # current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset = np.array([0, 0, 0.25]),
                end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi, theta * 2 * np.pi / 360]))
            )                    
            # target object를 찾기 위한 controller 동작
            # controller의 동작이 끝났을 때, detection을 수행
            if my_controller.is_done():
                # camera에서부터 rgb, distance 이미지 획득
                rgb_image = camera.get_rgba()[:, :, :3]
                rgb_image.save(os.path.join(save_root, f'rgb_image_{ep_num}.png'))
                # RGB카메라에서 물체의 Bounding Box가 있는지 확인
                bbox = camera.get_current_frame()["bounding_box_2d_tight"]
                print('bbox', bbox)

                # detection이 끝난 후, controller reset 및 while문 나가기
                my_controller.reset()
                break
            
            # 선언한 action을 입력받아 articulation_controller를 통해 action 수행
            # Controller 내부에서 계산된 joint position값을 통해 action을 수행함
            articulation_controller.apply_action(actions)
                
    # 이전에 선언한 found_obj를 확인하여 True인 경우(물체를 찾는 경우), 
    # pick place를 하기 위해 for문 나가기
    if found_obj:
        print('found object')
        break
