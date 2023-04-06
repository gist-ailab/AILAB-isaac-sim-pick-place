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
from omni.isaac.sensor import Camera
import carb
import sys
import numpy as np
import argparse
import os


import matplotlib.pyplot as plt
# import cv2
from PIL import Image

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
        position=np.array([0.1, 0.33, 0.1 + size_scale/2]),
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

def capture_img(mode, rgb_camera, depth_camera, num):
    if mode == 'rgb':
        rgb_image = rgb_camera.get_rgba()[:, :, :3]
        depth_image = rgb_camera.get_current_frame()["distance_to_camera"]
        instance_segmentation_image = rgb_camera.get_current_frame()["instance_segmentation"]["data"]
        bbox = rgb_camera.get_current_frame()["bounding_box_2d_tight"]
    elif mode == 'd':
        rgb_image = depth_camera.get_rgba()[:, :, :3]
        depth_image = depth_camera.get_current_frame()["distance_to_camera"]
        instance_segmentation_image = depth_camera.get_current_frame()["instance_segmentation"]["data"]
        bbox = depth_camera.get_current_frame()["bounding_box_2d_tight"]
    
    ## post process segmentation mask
    np.subtract(instance_segmentation_image, 1, out=instance_segmentation_image, where=instance_segmentation_image!=0)
    
    ## save image
    rgb_image = Image.fromarray((rgb_image).astype(np.uint8))
    rgb_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_{num}_rgb.png')
    instance_segmentation_image = Image.fromarray((instance_segmentation_image*255).astype(np.uint8))
    instance_segmentation_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_{num}_mask.png')
    depth_image = Image.fromarray((depth_image * 255.0).astype(np.uint8))
    depth_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_{num}_d.png')

    return len(np.unique(instance_segmentation_image)) -1, bbox


found_cube = False
z = 0.3
qw, qx, qy, qz = 0, 0, 0, 0.25-size_scale_z/2  # 0.707, 0, 0, 0.707

## find object and reach it
for r in range(3, 10, 2):
    for theta in range (0, 360, 45):
        x, y = r/10 * np.cos(theta), r/10 * np.sin(theta)
        print('[', r, ']', round(x,1), round(y,1))
        while simulation_app.is_running():
            my_world.step(render=True)
            if my_world.is_playing():
                observations = my_world.get_observations()
                actions = my_controller2.forward(
                    # picking_position=cube.get_local_pose()[0],
                    picking_position=np.array([x, y, z]),
                    placing_position=np.array([0.4, -0.33, 0.02]),
                    current_joint_positions=my_ur5e.get_joint_positions(),
                    end_effector_offset=np.array([0, 0, 0.25-size_scale_z/2]),
                )
                if my_controller2.is_done():
                    print('done reaching target')
                    found_cube, bbox = capture_img('rgb', rgb_camera, depth_camera, str(r) + '_' + str(theta))
                    my_controller2.reset()
                    break
                articulation_controller.apply_action(actions)
            if args.test is True:
                break
        if found_cube:
            print('found cube')
            print('bbox', bbox)
            while simulation_app.is_running():
                if 400 < bbox['data'][0][1] < 1520 and 400 < bbox['data'][0][2] < 680:
                    print('bbox is in the center')
                    break
                elif bbox['data'][0][1] <= 400:
                    print('bbox is on the right')
                    qx += 0.1
                    found_cube = False
                elif bbox['data'][0][1] >= 1520:
                    print('bbox is on the left')
                    qx -= 0.1
                    found_cube = False
                elif bbox['data'][0][2] <= 400:
                    print('bbox is on the top')
                    qy += 0.1
                    found_cube = False
                elif bbox['data'][0][2] >= 680:
                    print('bbox is on the bottom')
                    qy -= 0.1
                    found_cube = False
                # if 400 < bbox['data'][0][1] < 1520 and 450 < bbox['data'][0][2] < 630:
                #     print('bbox is in the center')
                #     break
                # elif bbox['data'][0][1] <= 400:
                #     print('bbox is on the right')
                #     qx += 0.1
                #     found_cube = False
                # elif bbox['data'][0][1] >= 1520:
                #     print('bbox is on the left')
                #     qx -= 0.1
                #     found_cube = False
                # elif bbox['data'][0][2] <= 450:
                #     print('bbox is on the top')
                #     qy += 0.1
                #     found_cube = False
                # elif bbox['data'][0][2] >= 630:
                #     print('bbox is on the bottom')
                #     qy -= 0.1
                #     found_cube = False
            
                my_world.step(render=True)
                if my_world.is_playing():
                    observations = my_world.get_observations()
                    actions = my_controller2.forward(
                        # picking_position=cube.get_local_pose()[0],
                        picking_position=np.array([x, y, z]),
                        placing_position=np.array([0.4, -0.33, 0.02]),
                        current_joint_positions=my_ur5e.get_joint_positions(),
                        end_effector_offset=np.array([qw, qx, qy, qz]),
                    )
                    found_cube, bbox = capture_img('rgb', rgb_camera, depth_camera, str(r) + '_' + str(theta))

                    if my_controller2.is_done():
                        print('done reaching target')
                        found_cube, bbox = capture_img('rgb', rgb_camera, depth_camera, str(r) + '_' + str(theta))
                        print('bbox', bbox)
                        my_controller2.reset()
                        break
                    articulation_controller.apply_action(actions)
            break
    if found_cube:
        break


print('#########')
print('#########')
print('#########')
x, y = float((780-300+317)/1080), float((714-300+42)/1920)
z = 0.235
print(x, y, z)
print(cube.get_local_pose()[0])
while simulation_app.is_running():
    print('pick and place')
    my_world.step(render=True)
    if my_world.is_playing():
        observations = my_world.get_observations()
        actions = my_controller.forward(
            picking_position=cube.get_local_pose()[0],
            # picking_position=np.array([x, y, z]),
            placing_position=np.array([0.4, -0.33, 0.02]),
            current_joint_positions=my_ur5e.get_joint_positions(),
            end_effector_offset=np.array([0, 0, 0.25-size_scale_z/2]),
        )
        found_cube, bbox = capture_img('rgb', rgb_camera, depth_camera, str(r) + '_' + str(theta))

        if my_controller.is_done():
            print('done reaching target')
            found_cube, bbox = capture_img('rgb', rgb_camera, depth_camera, str(r) + '_' + str(theta))
            print('bbox', bbox)
            my_controller.reset()
            break
        articulation_controller.apply_action(actions)




simulation_app.close()
