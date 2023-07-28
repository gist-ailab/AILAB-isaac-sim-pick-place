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

# World 생성
my_world = World(stage_units_in_meters=1.0)

# Task 생성
my_task = UR5ePickPlace()
my_world.add_task(my_task)
my_world.reset()

# Controller 생성
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = BasicManipulationController(
    # Controller의 이름 설정
    name='basic_manipulation_controller',
    # 로봇 모션 controller 설정
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_ur5e, attach_gripper=True
    ),
    # 로봇의 gripper 설정
    gripper=my_ur5e.gripper,
    # phase의 진행 속도 설정
    events_dt=[0.008],
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()
my_controller.reset()

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# 시뮬레이션 앱 실행 후 dalay를 위한 변수
max_step = 150

# 시뮬레이션 앱이 실행 중이면 동작
ee_target_position = np.array([0.25, -0.23, 0.4]) 

while simulation_app.is_running():
    # 생성한 world 에서 physics simulation step​
    my_world.step(render=True)

    if my_world.is_playing():
        if my_world.current_time_step_index > max_step:
            # my_world로 부터 observation 값들 획득​
            observations = my_world.get_observations()

            # 획득한 observation을 pick place controller에 전달
            actions = my_controller.forward(
                target_position=ee_target_position,
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset = np.array([0, 0, 0.14])
            )

            # controller의 동작이 끝났음을 출력
            if my_controller.is_done():
                print("done position control of end-effector")
                break
            
            # 컨트롤러 내부에서 계산된 타겟 joint position값을
            # articulation controller에 전달하여 action 수행
            articulation_controller.apply_action(actions)
            
# 시뮬레이션 종료
simulation_app.close()