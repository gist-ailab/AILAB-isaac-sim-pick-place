# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from tasks.pick_place_task import UR5ePickPlace
from controllers.pick_place_controller_robotiq import PickPlaceController
from reach_target_controller import ReachTargetController
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_camera_path
from omni.isaac.sensor import Camera
from correct_radial_distortion import depth_image_from_distance_image
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import glob, os, random
import matplotlib.pyplot as plt


working_dir = os.path.dirname(os.path.realpath(__file__))
# objects_path = os.path.join(working_dir, "ycb_usd/*/*.usd")
objects_path = os.path.join(working_dir, "/home/nam/workspace/ycb_usd/ycb/*/*.usd")
objects_list = glob.glob(objects_path)
objects_list = random.sample(objects_list, 3)
# get three objects randomly

objects_position = np.array([[0.3, 0.3, 0.1],
                             [-0.3, 0.3, 0.1],
                             [-0.3, -0.3, 0.1]])
offset = np.array([0, 0, 0.1])
target_position = np.array([0.4, -0.33, 0.55])  # 0.55 for considering the length of the gripper tip
target_orientation = np.array([0, 0, 0, 1])

my_world = World(stage_units_in_meters=1.0)
my_task = UR5ePickPlace(objects_list = objects_list,
                        objects_position = objects_position,
                        offset=offset)  # releasing offset at the target position
my_world.add_task(my_task)
my_world.reset()
task_params = my_task.get_params()
my_ur5 = my_world.scene.get_object(task_params["robot_name"]["value"])
my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_ur5.gripper, robot_articulation=my_ur5
    )
my_controller2 = ReachTargetController(
    name="reach_controller", gripper=my_ur5.gripper, robot_articulation=my_ur5, end_effector_initial_height=0.3
)
articulation_controller = my_ur5.get_articulation_controller()

# viewport = get_active_viewport()
# viewport.set_active_camera('/World/ur5e/realsense/Depth')
# viewport.set_active_camera('/OmniverseKit_Persp')

depth_camera = Camera(
    prim_path="/World/UR5e/realsense/Depth",
    frequency=20,
    resolution=(1920, 1080),
)
depth_camera.initialize()
depth_camera.add_distance_to_camera_to_frame()
depth_camera.add_instance_segmentation_to_frame()
depth_camera.add_instance_id_segmentation_to_frame()
depth_camera.add_semantic_segmentation_to_frame()
depth_camera.add_bounding_box_2d_loose_to_frame()
depth_camera.add_bounding_box_2d_tight_to_frame()

r, theta, z = 4, 0, 0.3
found_obj = False
print('reach-target')
for theta in range(0, 360, 45):
    x, y = r/10 * np.cos(theta/360*2*np.pi), r/10 * np.sin(theta/360*2*np.pi)
    print('[', r, ']', round(x,1), round(y,1))
    while simulation_app.is_running():
        my_world.step(render=True)
        if my_world.is_playing():
            observations = my_world.get_observations()
            actions = my_controller2.forward(
                # picking_position=cube.get_local_pose()[0],
                picking_position=np.array([x, y, z]),
                current_joint_positions=my_ur5.get_joint_positions(),
                # end_effector_offset=np.array(([0.125, 0.095, 0.04])),
                end_effector_offset=np.array([0, 0, 0.25]),
                theta=theta
            )
            if my_controller2.is_done():
                rgb_image = depth_camera.get_rgba()[:, :, :3]
                depth_image = depth_camera.get_current_frame()["distance_to_camera"]
                instance_segmentation_image = depth_camera.get_current_frame()["instance_segmentation"]["data"]
                tight_bbox = depth_camera.get_current_frame()["bounding_box_2d_tight"]
                
                if {'class': 'object1'} in tight_bbox["info"]["idToLabels"].values():
                    depth_camera_intrinsics = depth_camera.get_intrinsics_matrix()
                    n_depth_image = depth_image_from_distance_image(depth_image, depth_camera_intrinsics)
                                    
                    imgplot = plt.imshow(rgb_image)
                    plt.show()
                    inssegplot = plt.imshow(instance_segmentation_image)
                    plt.show()
                    ndepthplot = plt.imshow(n_depth_image)
                    plt.show()
                    
                    bbox_info = tight_bbox["info"]["bboxIds"]
                    bboxes = {}
                    for i in range(len(bbox_info)):
                        id = bbox_info[i]
                        bboxes["obj"+str(id)] = tight_bbox["data"][int(id)]
                    
                    bbox = bboxes["obj"+str(1)]
                    center = [int((bbox[1]+bbox[3])/2), int((bbox[2]+bbox[4])/2)]
                    depth = depth_image[center[0]][center[1]]
                    center = np.expand_dims(center, axis=0)
                    world_center = depth_camera.get_world_points_from_image_coords(center, depth)
                    
                    # angle, length, width, center = inference_ggcnn(rgb_image, n_depth_image, bboxes["obj"+str(1)])
                    # center = np.array(center)
                    # depth = n_depth_image[center[1]][center[0]]
                    
                    # center = np.expand_dims(center, axis=0)
                    # world_center = depth_camera.get_world_points_from_image_coords(center, depth)
                    # print(world_center)
                    # print(center)
                    # print(angle)
                    # angle = np.arctan(world_center[0][1]/world_center[0][0])
                    angle = theta * 2 * np.pi / 360
                    found_obj = True
                    
                my_controller2.reset()
                break
            articulation_controller.apply_action(actions)
    if found_obj:
        print('found obj')
        break

i = 0
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            my_controller.reset()
        observations = my_world.get_observations()
        actions = my_controller.forward(
            # picking_position=observations[task_params["task_object_name_0"]["value"]]["position"],
            # placing_position=observations[task_params["task_object_name_0"]["value"]]["target_position"],
            # current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
            picking_position=np.array([world_center[0][0], world_center[0][1], 0.01]),
            placing_position=np.array([0.4, -0.33, 0.02]),
            current_joint_positions=my_ur5.get_joint_positions(),
            end_effector_offset=np.array([0.125, 0.095, 0.04]),
            # end_effector_offset=np.array([0, 0, 0.04]),
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, angle])),
        )
        if my_controller.is_done():
            print("done picking and placing")
        articulation_controller.apply_action(actions)
simulation_app.close()
