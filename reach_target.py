# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.semantics import add_update_semantics
from pick_place_controller import PickPlaceController
from reach_target_controller import ReachTargetController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.sensor import Camera
import carb
import sys
import numpy as np
import argparse
import os


import matplotlib.pyplot as plt
import cv2
from correct_radial_distortion import depth_image_from_distance_image
from ggcnn.inferece_ggcnn import inference_ggcnn
# from inference_ggcnn import inference_ggcnn
from PIL import Image
import random
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

ur5e_usd_path = "/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/ur5e_handeye_gripper.usd"
# ur5e_usd_path = "/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/ur5e_handeye_gripper.usd"
if os.path.isfile(ur5e_usd_path):
    pass
else:
    raise Exception(f"{ur5e_usd_path} not found")

add_reference_to_stage(usd_path=ur5e_usd_path, prim_path="/World/UR5e")
gripper = ParallelGripper(
    end_effector_prim_path="/World/UR5e/right_inner_finger_pad",
    joint_prim_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0.0, 0.0]),
    joint_closed_positions=np.array([np.pi*2/9, -np.pi*2/9]),
    action_deltas=np.array([-np.pi*2/9, np.pi*2/9]),
)
my_ur5e = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/UR5e", name="my_ur5e", end_effector_prim_name="right_inner_finger_pad", gripper=gripper
    )
)
# my_ur5e.set_joints_default_state(
#     positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
# )
rgb_camera = Camera(
    prim_path="/World/UR5e/realsense/RGB",
    frequency=20,
    resolution=(1920, 1080),
)
depth_camera = Camera(
    prim_path="/World/UR5e/realsense/Depth",
    frequency=20,
    resolution=(1920, 1080),
)

print('rgb camera intrinsic\n', rgb_camera.get_intrinsics_matrix())
print('depth camera intrinsic\n', depth_camera.get_intrinsics_matrix())
size_scale = 0.03
size_scale_z = 0.03
cube = my_world.scene.add(
    DynamicCuboid(
        name="cube",
        # position=np.array([5, 5, 0.1 + size_scale/2]),
        position=np.array([0.3, 0.33, 0.1 + size_scale/2]),
        prim_path="/World/Cube",
        scale=np.array([size_scale, size_scale, size_scale]),
        size=1.0,
        color=np.array([1, 0, 1]),
        mass=0
    )
)
my_world.scene.add_default_ground_plane()
my_ur5e.gripper.set_default_state(my_ur5e.gripper.joint_opened_positions)
my_world.reset()

rgb_camera.initialize()
rgb_camera.add_distance_to_camera_to_frame()
rgb_camera.add_instance_segmentation_to_frame()
rgb_camera.add_instance_id_segmentation_to_frame()
rgb_camera.add_semantic_segmentation_to_frame()
rgb_camera.add_bounding_box_2d_loose_to_frame()
rgb_camera.add_bounding_box_2d_tight_to_frame()

depth_camera.initialize()
depth_camera.add_distance_to_camera_to_frame()
depth_camera.add_instance_segmentation_to_frame()
depth_camera.add_instance_id_segmentation_to_frame()
depth_camera.add_semantic_segmentation_to_frame()
depth_camera.add_bounding_box_2d_loose_to_frame()
depth_camera.add_bounding_box_2d_tight_to_frame()

stage = get_current_stage()
cube_prim = stage.DefinePrim("/World/Cube")
add_update_semantics(prim=cube_prim, semantic_label="cube")

my_world.reset()

my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e, end_effector_initial_height=0.3
)
my_controller2 = ReachTargetController(
    name="reach_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e, end_effector_initial_height=0.3
)
articulation_controller = my_ur5e.get_articulation_controller()



## --------------------------   Camera Capture Function   -------------------------- ##
def capture_img(mode, camera, camera_intrinsics, num):

    rgb_image = camera.get_rgba()[:, :, :3]
    depth_image = camera.get_current_frame()["distance_to_camera"]
    instance_segmentation_image = camera.get_current_frame()["instance_segmentation"]["data"]
    bbox = camera.get_current_frame()["bounding_box_2d_tight"]
    
    ## post process segmentation mask
    np.subtract(instance_segmentation_image, 1, out=instance_segmentation_image, where=instance_segmentation_image!=0)
    
    ## post process depth image
    n_depth_image = depth_image_from_distance_image(depth_image, camera_intrinsics)

    ## save image
    rgb_image = Image.fromarray((rgb_image).astype(np.uint8))
    instance_segmentation_image = Image.fromarray((instance_segmentation_image*255).astype(np.uint8))
    n_depth_image = Image.fromarray((n_depth_image * 255.0).astype(np.uint8))
    # rgb_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_{num}_rgb.png')
    # instance_segmentation_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_{num}_mask.png')
    # n_depth_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_{num}_d.png')

    return len(np.unique(instance_segmentation_image)) -1,      \
           bbox,    \
           cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR),    \
           cv2.cvtColor(np.array(n_depth_image), cv2.COLOR_RGB2BGR),    \
           cv2.cvtColor(np.array(instance_segmentation_image), cv2.COLOR_RGB2BGR)


## --------------------------   1. Spawn Objects   -------------------------- ##
# objects = glob.glob("/ailab_mat/dataset/ycb_usd/ycb/*/*.usd")
objects = glob.glob("/home/nam/workspace/ycb_usd/ycb/*/*.usd")

for l in range(3):
    size_z = (random.random()*0.09+0.03)*2
    size = size_z
    pos_y = (random.random()*1.2+0.8)
    pos_x = (random.random()*1.2+0.8)
    # pos_x = (random.random()*1.2-0.4)
    # pos_z = size_z
    position=np.array([pos_x, pos_y, 0.1 + size_z/2])
    a = random.randint(0, len(objects)-1)
    create_prim(usd_path=objects[a], prim_path="/World/object"+str(l), position=position, scale=[0.2,0.2,0.2])


## --------------------------   2. Find a target   -------------------------- ##

found_cube = False
z = 0.4
qw, qx, qy, qz = 0, 0, 0, 0.25-size_scale_z/2  # 0.707, 0, 0, 0.707
mode = 'rgb'

if mode == 'rgb':
    camera = rgb_camera
    camera_intrinsics = rgb_camera.get_intrinsics_matrix()
elif mode == 'd' or mode == 'depth':
    camera = depth_camera
    camera_intrinsics = depth_camera.get_intrinsics_matrix()

## find object and reach it
r, theta = 5, 30

x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
print('[', r, ']', round(x,1), round(y,1))
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        observations = my_world.get_observations()
        actions = my_controller2.forward(
            # picking_position=cube.get_local_pose()[0],
            picking_position=np.array([x, y, z]),
            current_joint_positions=my_ur5e.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.25-size_scale_z/2]),
            theta=theta,
        )
        if my_controller2.is_done():
            print('done reaching target')
            found_cube, bbox, rgb, depth, mask = capture_img(mode, camera, camera_intrinsics, str(r) + '_' + str(theta))
            my_controller2.reset()
            break
        articulation_controller.apply_action(actions)
    if args.test is True:
        break
if found_cube:
    print('found cube')
    print('bbox', bbox)


simulation_app.close()