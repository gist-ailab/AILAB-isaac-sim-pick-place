
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.kit.viewport.utility import get_active_viewport

from pick_place_controller import PickPlaceController # -> from utils.controllers.pick_place_controller_robotiq import PickPlaceController
from utils.controllers.end_effector_controller import EndEffectorController
from omni.isaac.universal_robots.controllers import RMPFlowController
from inference_ggcnn import inference_ggcnn

import sys
sys.path.append('/isaac-sim/exts/omni.isaac.examples/')
from omni.isaac.examples.ailab_script import AILabExtension
from omni.isaac.examples.ailab_examples import AILab

from train_model import get_model_object_detection
from depth_to_distance import depth_image_from_distance_image

import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
import torch
from pathlib import Path

from utils.tasks.pick_place_vision_task import UR5ePickPlace
import coco.transforms as T

# set gui extension
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

# get ycb directories to sys.path
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
        'label': obj_idx, # set object label 2 ~ 
    }
    label2name[obj_idx]=os.path.basename(obj_dir)

objects_list = random.sample(list(object_info.values()), 3)
objects_usd_list = []
for obj_info in objects_list:
    objects_usd_list.append(obj_info['usd_file'])

# if you don't declare objects_position, the objects will be placed randomly
objects_position = np.array([[0.5, 0.2, 0.1],
                             [-0.6, 0.3, 0.1],
                             [-0.6, -0.3, 0.1]])
offset = np.array([0, 0, 0.1])  # releasing offset at the target position
target_position = np.array([0.4, -0.33, 0.55])  # 0.55 for considering the length of the gripper tip
target_orientation = np.array([0, 0, 0, 1])

my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace(objects_list = objects_usd_list,
                        objects_position = objects_position,
                        offset=offset)  # releasing offset at the target position

my_world.add_task(my_task)
my_world.reset()

task_params = my_task.get_params()
my_ur5e = my_world.scene.get_object(task_params["robot_name"]["value"])

camera = my_task.get_camera()

stage = get_current_stage()
for l in range(3):
    object_prim = stage.DefinePrim("/World/object"+str(l))
    add_update_semantics(prim=object_prim, semantic_label="object"+str(l))

my_world.reset()
my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e, end_effector_initial_height=0.3
)
my_controller2 = EndEffectorController(
    name='end_effector_controller',
    cspace_controller=RMPFlowController(
        name="end_effector_controller_cspace_controller", robot_articulation=my_ur5e, attach_gripper=True
    ),
    gripper=my_ur5e.gripper,
    events_dt=[0.008],
)
articulation_controller = my_ur5e.get_articulation_controller()


##########detection model load##############

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 43
model = get_model_object_detection(num_classes)
model.to(device)
model.load_state_dict(torch.load(os.path.join(Path(working_dir).parent, "checkpoint/model_99.pth")))
model.eval()

transforms = []
transforms.append(T.PILToTensor())
transforms.append(T.ConvertImageDtype(torch.float))
transforms = T.Compose(transforms)

######################################################

viewport = get_active_viewport()
viewport.set_active_camera('/World/ur5e/realsense/Depth')
viewport.set_active_camera('/OmniverseKit_Persp')

r, theta, z = 4, 0, 0.3
found_obj = False
print('reach-target')
for i in range(len(objects_list)):
    print("object_{}: {}".format(i, objects_list[i]['name']))
for theta in range(0, 360, 45):
    x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
                my_controller2.reset()
                    
            if gui_test.use_custom_updated:
                observations = my_world.get_observations()
                actions = my_controller2.forward(
                    picking_position=np.array([x, y, z]),
                    current_joint_positions=my_ur5e.get_joint_positions(),
                    end_effector_offset=np.array([0, 0, 0.25]),
                    end_effector_orientation=euler_angles_to_quat(np.array([0, np.pi, theta * 2 * np.pi / 360]))
                )
                if my_controller2.is_done():
                    rgb_image = camera.get_rgba()[:, :, :3]
                    distance_image = camera.get_current_frame()["distance_to_camera"]
                    
                    ##############detection inference######################
                    image = Image.fromarray(rgb_image)
                    image, _ = transforms(image=image, target=None)
                    with torch.no_grad():
                        prediction = model([image.to(device)])
                    prediction[0]['labels']=prediction[0]['labels'].cpu().detach().numpy()
                    labels_name = []
                    scores = []
                    for i in range(len(list(prediction[0]['boxes']))):
                        if prediction[0]['scores'][i]>0.9:
                            predict_label = prediction[0]['labels'][i]
                            labels_name.append(label2name[predict_label])
                        scores.append(prediction[0]['scores'][i])
                    
                    target = objects_list[int(gui_test.current_target.split('_')[-1])]['name']
                    
                    if target in labels_name:
                        found_obj = True
                        labels_name = np.array(labels_name)
                        indexes = np.where(labels_name == target)
                    
                        #######draw bbox in image############
                        image = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
                        draw = ImageDraw.Draw(image)
                        for i in range(len(list(prediction[0]['boxes']))):
                            if prediction[0]['scores'][i]>0.9:
                                predict_label = prediction[0]['labels'][i]
                                draw.multiline_text((list(prediction[0]['boxes'][i])), text = label2name[predict_label])
                                draw.rectangle((list(prediction[0]['boxes'][i])), outline=(1,0,0),width=5)
                        image = np.array(image)
                        plt.imshow(image)
                        plt.show()
                                        
                        #########inference ggcnn##############3
                        camera_intrinsics = camera.get_intrinsics_matrix()
                        depth_image = depth_image_from_distance_image(distance_image, camera_intrinsics)
                        
                        target_scores = []
                        for index in indexes:
                            target_scores.append(scores[index[0]])
                        max_score = max(target_scores)
                        idx = scores.index(max_score)
                        bbox = prediction[0]['boxes'][idx]
                        
                        ggcnn_angle, length, width, center = inference_ggcnn(
                            rgb=rgb_image, depth=depth_image, bbox=bbox)
                        center = np.array(center)
                        depth = distance_image[center[1]][center[0]]
                        
                        center = np.expand_dims(center, axis=0)
                        world_center = camera.get_world_points_from_image_coords(center, depth)
                        angle = theta * 2 * np.pi / 360 + ggcnn_angle
                        print("world_center: {}, length: {}, width: {}, angle: {}".format(world_center, length, width, angle))
                        print("object_position: {}".format(observations[task_params["task_object_name_0"]["value"]]["position"]))
                                                
                    my_controller2.reset()
                    break
                articulation_controller.apply_action(actions)
                
    if found_obj:
        print('found object')
        break

print('pick-and-place')
change_world_center = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        
        observations = my_world.get_observations()              
        actions = my_controller.forward(
            picking_position=np.array([world_center[0][0], world_center[0][1], 0.01]),
            placing_position=observations[task_params[gui_test.current_target]["value"]]["target_position"],
            current_joint_positions=my_ur5e.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.25]),
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, angle])),
        )
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)

simulation_app.close()