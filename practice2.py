# import numpy as np
# import cv2
# import matplotlib.pyplot as plt



# def depth_image_from_distance_image(distance, intrinsics):
#     """Computes depth image from distance image.
    
#     Background pixels have depth of 0
    
#     Args:
#         distance: HxW float array (meters)
#         intrinsics: 3x3 float array
    
#     Returns:
#         z: HxW float array (meters)
    
#     """
#     fx = intrinsics[0][0]
#     cx = intrinsics[0][2]
#     fy = intrinsics[1][1]
#     cy = intrinsics[1][2]
    
#     height, width = distance.shape
#     xlin = np.linspace(0, width - 1, width)
#     ylin = np.linspace(0, height - 1, height)
#     px, py = np.meshgrid(xlin, ylin)
    
#     x_over_z = (px - cx) / fx
#     y_over_z = (py - cy) / fy
    
#     z = distance / np.sqrt(1. + x_over_z**2 + y_over_z**2)
#     return z




# depth_path = "/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_d.png"
# new_depth_path = "/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgbcam_newd.png"
# depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
# # cv2.resize(depth, (1920, 1080))
# print(type(depth))
# print(depth.size)   # 1920*1080*3       6220800 # 1920*1080*1       2073600
# print(depth.shape)  # (1080, 1920)
# # intrinsics = [[940.50757, 0.0, 960.0], [0.0, 529.0355, 540.0], [0.0, 0.0, 1.0]]         # depth camera intrinsics
# intrinsics = [[1398.3395, 0.0, 960.0], [0.0, 786.56598, 540.0], [0.0, 0.0, 1.0]]        # rgb camera intrinsics



# new_depth = np.uint16(depth_image_from_distance_image(depth, intrinsics))
# plt.imshow(depth)
# plt.show()

# plt.imshow(new_depth)
# plt.show()
# cv2.imwrite(new_depth_path, new_depth*255)


# ## =========================================================== ##
# import cv2
# import pyrealsense2 as rs

# rgb = cv2.imread("/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/rgb_image_3.png")
# depth = cv2.imread("/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/depth_image_3_new.png", cv2.IMREAD_ANYDEPTH)
# depth_and_color_frameset = [rgb, depth]


# pipeline = rs.pipeline()
# config = rs.config()

# align = rs.align(rs.stream.color)
# aligned_frames = align.proccess(depth_and_color_frameset)
# color_frame = aligned_frames.first(rs.stream.color)
# aligned_depth_frame = aligned_frames.get_depth_frame()

# ## =========================================================== ##

# while simulation_app.is_running():
#     my_world.step(render=True)
#     if my_world.is_playing():
#         if my_world.current_time_step_index == 0:
#             my_world.reset()
#             my_controller.reset()
#         # elif my_world.current_time_step_index == 200:
#         if my_controller._event == 0:
#             if my_controller._t >= 0.99:
#             # if my_world.current_time_step_index > 110:
#                 my_controller.pause()
#                 # print(my_controller._t)
#                 # print(my_world.current_time_step_index)

#                 capture_img('rgb', rgb_camera, depth_camera, my_controller._t)
                
#                 my_controller.resume()
#         # elif my_world.current_time_step_index > 210:
#         observations = my_world.get_observations()
#         actions = my_controller.forward(
#             picking_position=cube.get_local_pose()[0],
#             # picking_position=np.array([0.5, 1-0.842, 0.02]),
#             placing_position=np.array([0.4, -0.33, 0.02]),
#             current_joint_positions=my_ur5e.get_joint_positions(),
#             end_effector_offset=np.array([0, 0, 0.25-size_scale_z/2]),
#         )
#         if my_controller.is_done():
#             print("done picking and placing")
#         articulation_controller.apply_action(actions)
#     if args.test is True:
#         break

# ## =========================================================== ##
# ### BACKUP CODE ###

# # Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# #
# # NVIDIA CORPORATION and its licensors retain all intellectual property
# # and proprietary rights in and to this software, related documentation
# # and any modifications thereto.  Any use, reproduction, disclosure or
# # distribution of this software and related documentation without an express
# # license agreement from NVIDIA CORPORATION is strictly prohibited.
# #

# from omni.isaac.kit import SimulationApp

# simulation_app = SimulationApp({"headless": False})

# from omni.isaac.core import World
# from omni.isaac.manipulators import SingleManipulator
# from omni.isaac.manipulators.grippers import ParallelGripper
# from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
# from omni.isaac.core.utils.semantics import add_update_semantics
# from pick_place_controller import PickPlaceController
# from omni.isaac.core.objects import DynamicCuboid
# from omni.isaac.sensor import Camera
# import carb
# import sys
# import numpy as np
# import argparse
# import os


# import matplotlib.pyplot as plt
# # import cv2
# from PIL import Image

# parser = argparse.ArgumentParser()
# parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
# args, unknown = parser.parse_known_args()

# my_world = World(stage_units_in_meters=1.0)
# my_world.scene.add_default_ground_plane()

# # ur5e_usd_path = "/home/nam/.local/share/ov/pkg/isaac_sim-2022.2.0/workspace/ur5e_handeye_gripper.usd"
# ur5e_usd_path = "/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/ur5e_handeye_gripper.usd"
# if os.path.isfile(ur5e_usd_path):
#     pass
# else:
#     raise Exception(f"{ur5e_usd_path} not found")

# add_reference_to_stage(usd_path=ur5e_usd_path, prim_path="/World/UR5e")
# gripper = ParallelGripper(
#     end_effector_prim_path="/World/UR5e/right_inner_finger_pad",
#     joint_prim_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
#     joint_opened_positions=np.array([0.0, 0.0]),
#     joint_closed_positions=np.array([np.pi*2/9, -np.pi*2/9]),
#     action_deltas=np.array([-np.pi*2/9, np.pi*2/9]),
# )
# my_ur5e = my_world.scene.add(
#     SingleManipulator(
#         prim_path="/World/UR5e", name="my_ur5e", end_effector_prim_name="right_inner_finger_pad", gripper=gripper
#     )
# )
# # my_ur5e.set_joints_default_state(
# #     positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
# # )
# rgb_camera = Camera(
#     prim_path="/World/UR5e/realsense/RGB",
#     frequency=20,
#     resolution=(1920, 1080),
# )
# depth_camera = Camera(
#     prim_path="/World/UR5e/realsense/Depth",
#     frequency=20,
#     resolution=(1920, 1080),
# )

# print('rgb camera intrinsic\n', rgb_camera.get_intrinsics_matrix())
# print('depth camera intrinsic\n', depth_camera.get_intrinsics_matrix())
# size_scale = 0.03
# size_scale_z = 0.03
# cube = my_world.scene.add(
#     DynamicCuboid(
#         name="cube",
#         position=np.array([0.4, 0.33, 0.1 + size_scale/2]),
#         prim_path="/World/Cube",
#         scale=np.array([size_scale, size_scale, size_scale]),
#         size=1.0,
#         color=np.array([1, 0, 1]),
#         mass=0
#     )
# )
# my_world.scene.add_default_ground_plane()
# my_ur5e.gripper.set_default_state(my_ur5e.gripper.joint_opened_positions)
# my_world.reset()

# rgb_camera.initialize()
# rgb_camera.add_distance_to_camera_to_frame()
# rgb_camera.add_instance_segmentation_to_frame()
# rgb_camera.add_instance_id_segmentation_to_frame()
# rgb_camera.add_semantic_segmentation_to_frame()
# rgb_camera.add_bounding_box_2d_loose_to_frame()
# rgb_camera.add_bounding_box_2d_tight_to_frame()

# depth_camera.initialize()
# depth_camera.add_distance_to_camera_to_frame()
# depth_camera.add_instance_segmentation_to_frame()
# depth_camera.add_instance_id_segmentation_to_frame()
# depth_camera.add_semantic_segmentation_to_frame()
# depth_camera.add_bounding_box_2d_loose_to_frame()
# depth_camera.add_bounding_box_2d_tight_to_frame()

# stage = get_current_stage()
# cube_prim = stage.DefinePrim("/World/Cube")
# add_update_semantics(prim=cube_prim, semantic_label="cube")

# my_world.reset()

# my_controller = PickPlaceController(
#     name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e, end_effector_initial_height=0.3
# )
# articulation_controller = my_ur5e.get_articulation_controller()

# def capture_img(mode, rgb_camera, depth_camera, num):
#     if mode == 'rgb':
#         rgb_image = rgb_camera.get_rgba()[:, :, :3]
#         depth_image = rgb_camera.get_current_frame()["distance_to_camera"]
#         instance_segmentation_image = rgb_camera.get_current_frame()["instance_segmentation"]["data"]
#         loose_bbox = rgb_camera.get_current_frame()["bounding_box_2d_loose"]
#         tight_bbox = rgb_camera.get_current_frame()["bounding_box_2d_tight"]
#     elif mode == 'd':
#         rgb_image = depth_camera.get_rgba()[:, :, :3]
#         depth_image = depth_camera.get_current_frame()["distance_to_camera"]
#         instance_segmentation_image = depth_camera.get_current_frame()["instance_segmentation"]["data"]
#         loose_bbox = depth_camera.get_current_frame()["bounding_box_2d_loose"]
#         tight_bbox = depth_camera.get_current_frame()["bounding_box_2d_tight"]
    
#     ## post process segmentation mask
#     np.subtract(instance_segmentation_image, 1, out=instance_segmentation_image, where=instance_segmentation_image!=0)
    
#     ## save image
#     rgb_image = Image.fromarray((rgb_image).astype(np.uint8))
#     rgb_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_rgb_{num}.png')
#     instance_segmentation_image = Image.fromarray((instance_segmentation_image*255).astype(np.uint8))
#     instance_segmentation_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_mask_{num}.png')
#     depth_image = Image.fromarray((depth_image * 255.0).astype(np.uint8))
#     depth_image.save(f'/home/hse/.local/share/ov/pkg/isaac_sim-2022.2.0/isaac-sim-pick-place/data/{mode}cam_d_{num}.png')

#     print('loose bbox', loose_bbox)
#     print('tight bbox', tight_bbox)


# while simulation_app.is_running():
#     my_world.step(render=True)
#     if my_world.is_playing():
#         if my_controller._event == 0:
#             if my_controller._t < 0.1:
#                 my_controller.pause()
#                 capture_img('rgb', rgb_camera, depth_camera, my_controller._t)
#                 my_controller.resume()
#         # elif my_world.current_time_step_index > 210:
#         observations = my_world.get_observations()
#         actions = my_controller.forward(
#             picking_position=cube.get_local_pose()[0],
#             # picking_position=np.array([0.5, 1-0.842, 0.02]),
#             placing_position=np.array([0.4, -0.33, 0.02]),
#             current_joint_positions=my_ur5e.get_joint_positions(),
#             end_effector_offset=np.array([0, 0, 0.25-size_scale_z/2]),
#         )
#         if my_controller.is_done():
#             print("done picking and placing")
#         articulation_controller.apply_action(actions)
#     if args.test is True:
#         break


# simulation_app.close()









