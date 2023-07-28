
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.universal_robots.controllers import RMPFlowController



import numpy as np
import os, sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.controllers.basic_manipulation_controller import BasicManipulationController
from utils.tasks.pick_place_task import UR5ePickPlace

# World 생성
my_world = World(stage_units_in_meters=1.0)


# Task 생성
my_task = UR5ePickPlace()

# World에 Task 추가
my_world.add_task(my_task)
my_world.reset()

# Task로부터 ur5e와 camera를 획득
task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])

# PickPlace controller 생성
my_controller = BasicManipulationController(
    name='basic_manipulation_controller',
    cspace_controller=RMPFlowController(
        name="basic_manipulation_controller_cspace_controller", 
        robot_articulation=my_ur5e, 
        attach_gripper=True
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

ep_num = 0

# theta 값에 따라서 움직이며, target object를 찾을때까지 360도를 회전
# theta 값을 계속해서 달라지게 하기 위해 for문 사용(예제는 45도씩 회전하도록 하였음)
for theta in range(0, 360, 45):
    
    # theta 값에 따라서 end effector의 위치를 지정(x, y, z)
    r, z = 4, 0.3
    x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
    
    # 생성한 world 에서 physics simulation step​
    while simulation_app.is_running():
        
        my_world.step(render=True)
        if my_world.is_playing():
            
            # step이 0일때, world와 controller를 reset
            if my_world.current_time_step_index == 0:
                my_world.reset()
                my_controller.reset()
                
            # target object를 찾기 위한 controller 동작
            actions = my_controller.forward(
                target_position=np.array([x, y, z]),
                current_joint_positions=my_ur5e.get_joint_positions(),
                end_effector_offset = np.array([0, 0, 0.25]),
                end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi, theta * 2 * np.pi / 360]))
            )
            
            # controller reset 및 while문 나가기
            if my_controller.is_done():
                ep_num += 1
                print(ep_num)
                my_controller.reset()
                break
            
            # 선언한 action을 입력받아 articulation_controller를 통해 action 수행
            # Controller 내부에서 계산된 joint position값을 통해 action을 수행함
            articulation_controller.apply_action(actions)
        
        if ep_num == 8:
            print("done")
            break
        
simulation_app.close()
                