
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.kit.viewport.utility import get_active_viewport

import sys
sys.path.append('/isaac-sim/exts/omni.isaac.examples/')
from omni.isaac.examples.ailab_script import AILabExtension
from omni.isaac.examples.ailab_examples import AILab

from train_model import get_model_object_detection

import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from utils.controllers.basic_manipulation_controller import BasicManipulationController
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.tasks.pick_place_task import UR5ePickPlace
import coco.transforms as T

# set AILab GUI Extension
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


################ Rndom한 YCB 물체 3개를 생성을 포함하는 Task 생성 ########################

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

# 랜덤한 3개의 물체에 대한 usd file path 선택
objects_list = random.sample(list(object_info.values()), 3)
objects_usd_list = []
for obj_info in objects_list:
    objects_usd_list.append(obj_info['usd_file'])

# Random하게 생성된 물체들의 ​번호와 카테고리 출력 
for i in range(len(objects_list)):
    print("object_{}: {}".format(i, objects_list[i]['name']))

# 3개의 물체를 생성할 위치 지정(너무 멀어지는 경우 로봇이 닿지 않을 수 있음, 물체 사이의 거리가 가까울 경우 충돌이 발생할 수 있음)
objects_position = np.array([[0.5, 0, 0.1],
                             [-0.2, 0.5, 0.1],
                             [-0.55, 0.2, 0.1]])
offset = np.array([0, 0, 0.1])

# 물체를 놓을 위치(place position) 지정
target_position = np.array([0.4, -0.33, 0.55])
target_orientation = np.array([0, 0, 0, 1])

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성
my_task = UR5ePickPlace(objects_list = objects_usd_list,
                        objects_position = objects_position,
                        offset=offset)
                        
# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

####################################################################################

######################### Robot controller 생성 ####################################

# Task로부터 ur5e와 camera를 획득
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
camera = my_task.get_camera()

# PickPlace controller 생성
my_controller = PickPlaceController(
    name="pick_place_controller", 
    gripper=my_ur5e.gripper, 
    robot_articulation=my_ur5e
)

# EndEffector Controller 생성
my_controller2 = BasicManipulationController(
    name='end_effector_controller',
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", 
        robot_articulation=my_ur5e, 
        attach_gripper=True
    ),
    gripper=my_ur5e.gripper,
    events_dt=[0.008],
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()

####################################################################################

########################### Detection model load ###################################

# detection model load

# detection model input을 맞춰주기 위한 transform 생성

####################################################################################

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# 생성한 world 에서 physics simulation step​
while simulation_app.is_running():                      
    my_world.step(render=True)
                    