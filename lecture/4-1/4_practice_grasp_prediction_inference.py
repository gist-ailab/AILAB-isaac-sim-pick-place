
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


############### Random한 YCB 물체 3개를 생성을 포함하는 Task 생성 ########################

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

# 3개의 물체를 생성할 위치 지정(너무 멀어지는 경우 로봇이 닿지 않을 수 있음, 물체 사이의 거리가 가까울 경우 충돌이 발생할 수 있음)
objects_position = np.array([[0.5, 0, 0.1],
                             [-0.1, 0.5, 0.1],
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
my_controller2 = BasicManipulationController(
    name='basic_manipulation_controller',
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_ur5e, attach_gripper=True
    ),
    gripper=my_ur5e.gripper,
    events_dt=[0.008],
)

# robot control(PD control)을 위한 instance 선언
articulation_controller = my_ur5e.get_articulation_controller()

####################################################################################

########################### Detection model load ###################################

# detection model load
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 29
model = get_model_object_detection(num_classes)
model.to(device)
model.load_state_dict(torch.load(os.path.join(Path(working_dir).parent, "checkpoint/model_99.pth")))
model.eval()

# detection model input을 맞춰주기 위한 transform 생성
transforms = []
transforms.append(T.PILToTensor())
transforms.append(T.ConvertImageDtype(torch.float))
transforms = T.Compose(transforms)

####################################################################################

# GUI 상에서 보는 view point 지정(Depth 카메라 view에서 Perspective view로 변환시, 전체적으로 보기 편함)
viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

# target object를 찾았는지 확인하기 위해서 found_obj 선언(시작시에는 찾지 못한 상태이니 False)
found_obj = False
print('reach-target')

# theta 값에 따라서 움직이며, target object를 찾을때까지 360도를 회전
# theta 값을 계속해서 달라지게 하기 위해 for문 사용(예제는 45도씩 회전하도록 하였음)
for theta in range(0, 360, 45):
    # theta 값에 따라서 end effector의 위치를 지정(x, y, z)
    r, z = 4, 0.35
    x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
    
    # 생성한 world 에서 physics simulation step​
    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            
            # step이 0일때, world와 controller를 reset
            if my_world.current_time_step_index == 0:
                my_world.reset()
                my_controller2.reset()
                    
            # AILab Extention을 사용하여 target object를 지정해주었을때,
            # target object를 찾기 위한 controller 동작
            if gui_test.use_custom_updated:
                # 지정한 end effector 위치로 controller 동작
                actions = my_controller2.forward(
                    target_position=np.array([x, y, z]),
                    current_joint_positions=my_ur5e.get_joint_positions(),
                    end_effector_offset=np.array([0, 0, 0.14]),
                    end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi, theta * 2 * np.pi / 360]))
                )
                
                
########################### Detection model inference ###############################
                
                # controller의 동작이 끝났을 때, detection을 수행
                if my_controller2.is_done():
                    # camera에서부터 rgb, distance 이미지 획득
                    rgb_image = camera.get_rgba()[:, :, :3]
                    distance_image = camera.get_current_frame()["distance_to_camera"]
                    
                    # rgb 이미지를 detection model의 input에 맞게 transform
                    image = Image.fromarray(rgb_image)
                    image, _ = transforms(image=image, target=None)
                    
                    # detection model inference
                    with torch.no_grad():
                        prediction = model([image.to(device)])
                        
                    # Detection model의 출력을 object 카테고리 이름으로 변환 및 출력된 각 bbox의 score 값 확인
                    labels_name = []
                    scores = []
                    prediction[0]['labels']=prediction[0]['labels'].cpu().detach().numpy()
                    for i in range(len(list(prediction[0]['boxes']))):
                        if prediction[0]['scores'][i]>0.9:
                            predict_label = prediction[0]['labels'][i]
                            labels_name.append(label2name[predict_label])
                        scores.append(prediction[0]['scores'][i])
                    
                    # AILab Extention을 사용하여 지정된 target object의 카테고리 이름 찾기
                    target = objects_list[int(gui_test.current_target.split('_')[-1])]['name']
                    
                    # Detection 결과에 target object가 있는지 확인
                    if target in labels_name:
                        # 이전에 선언한 found_obj를 True로 바꿔서 target object를 찾은 것을 확인
                        found_obj = True
                        
                        # detection 결과 중, target object를 검출한 bbox의 index 확인
                        labels_name = np.array(labels_name)
                        indexes = np.where(labels_name == target)
                    
                        # Detection한 결과를 rgb 이미지 위에 그리기 ​
                        image = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
                        draw = ImageDraw.Draw(image)
                        for i in range(len(list(prediction[0]['boxes']))):
                            # 예측한 bbox의 score가 0.9 이상인 경우에 대해서 그리기
                            if prediction[0]['scores'][i]>0.9:
                                predict_label = prediction[0]['labels'][i]
                                draw.multiline_text((list(prediction[0]['boxes'][i])),
                                                     text = label2name[predict_label])
                                draw.rectangle((list(prediction[0]['boxes'][i])),
                                                outline=(1,0,0),width=5)
                        image = np.array(image)
                        plt.imshow(image)
                        plt.show()
                        
####################################################################################
                        
########################## Grasp prediction inference ###############################

                        # camera intrinsics을 이용하여 distance image를 depth image로 변환
                        
                        # Detection의 출력 중, target 물체에 대한 score가 가장 높은 bbox 선택 
                        target_scores = []
                        for index in indexes:
                            target_scores.append(scores[index[0]])
                        max_score = max(target_scores)
                        idx = scores.index(max_score)
                        bbox = prediction[0]['boxes'][idx]
                        
                        # GGCNN model inference
                        
                        # GGCNN에서 출력된 이미지 상의 center 값을 world coordinate으로 변환
                        
####################################################################################
                    
                    # detection과 grasp prediction이 끝난 후, controller reset 및 while문 나가기
                    my_controller2.reset()
                    break
                
                # 선언한 action을 입력받아 articulation_controller를 통해 action 수행
                # Controller 내부에서 계산된 joint position값을 통해 action을 수행함
                articulation_controller.apply_action(actions)
                
    # 이전에 선언한 found_obj를 확인하여 True인 경우(물체를 찾는 경우), 
    # pick place를 하기 위해 for문 나가기
    if found_obj:
        print('found object')
        break
    
# 이전 실습(only pick place)에서 사용했던 code와 거의 동일
print('pick-and-place')
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
            
        observations = my_world.get_observations()
        
        # picking position을 앞서 grasp prediction에서 얻는 center 값의 world coordinate으로 설정
        # end effector orientation을 앞서 grasp prediction을 통해 얻은 angle 값으로 설정
        actions = my_controller.forward(
            picking_position=np.array([world_center[0][0], world_center[0][1], 0.1218]),
            placing_position=observations[task_params[gui_test.current_target]["value"]]["target_position"],
            current_joint_positions=my_ur5e.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.14]),
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, angle])),
        )
        if my_controller.is_done():
            print("done picking and placing")
            break
        
        articulation_controller.apply_action(actions)

simulation_app.close()