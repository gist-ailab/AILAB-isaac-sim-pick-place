
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.kit.viewport.utility import get_active_viewport

import sys, os
from pathlib import Path
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from utils.tasks.pick_place_task import UR5ePickPlace

############### Random한 YCB 물체 생성을 포함하는 Task 생성 ######################

# YCB Dataset 물체들에 대한 정보 취득

# 랜덤한 물체에 대한 usd file path 선택

# Random하게 생성된 물체들의 ​번호와 카테고리 출력 

# 물체를 생성할 위치 지정(너무 멀어지는 경우 로봇이 닿지 않을 수 있음, 물체 사이의 거리가 가까울 경우 충돌이 발생할 수 있음)

# 물체를 놓을 위치(place position) 지정

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성

# World에 Task 추가

########################################################################

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# 생성한 world 에서 physics simulation step​
while simulation_app.is_running():
    my_world.step(render=True)
    
# simulation 종료​
simulation_app.close()