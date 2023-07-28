from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.rotations import euler_angles_to_quat

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.tasks.pick_place_task import UR5ePickPlace
from utils.controllers.RMPFflow_pickplace import RMPFlowController
from utils.controllers.basic_manipulation_controller import BasicManipulationController


############### 로봇의 기본적인 매니퓰레이션 동작을 위한 환경 설정 ################

# World 생성


# Task 생성


# World에 Task 추가 및 World 리셋



# Task로부터 로봇과 카메라 획득



# 로봇의 액션을 수행하는 Controller 생성










#########################################################################

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()


# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')


# target object를 찾기 위한 예제 코드
# end effector가 반지름 4를 가지며 theta가 45도씩 360도를 회전 수행
for theta in range(0, 360, 45):
    
    # theta 값에 따라서 end effector의 위치를 지정(x, y, z)
    r, z = 4, 0.55
    x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
    
    while simulation_app.is_running():
        # 생성한 world 에서 physics simulation step​
        my_world.step(render=True)
        if my_world.is_playing():
            
            # step이 0일때, World와 Controller를 reset
            if my_world.current_time_step_index == 0:
                my_world.reset()
                my_controller.reset()
                
############################# 로봇 액션 생성 ##############################
            # 획득한 observation을 pick place controller에 전달
            
            
            
            
            
            
            
#########################################################################
            
            # end effector가 원하는 위치에 도달하면
            # controller reset 및 while문 나가기
            if my_controller.is_done():
                my_controller.reset()
                break
            
############################# 로봇 액션 수행 ##############################
            # 선언한 action을 입력받아 articulation_controller를 통해 action 수행
            # Controller에서 계산된 joint position값을 통해 action을 수행함
            
#########################################################################
            
simulation_app.close()