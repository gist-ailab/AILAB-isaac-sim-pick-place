from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.universal_robots import UR10
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from franka.franka import Franka
from omni.isaac.sensor import Camera
from ur5 import UR5
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.syntheticdata import SyntheticData
import omni.replicator.core as rep

import numpy as np
import argparse
import matplotlib.pyplot as plt
import PIL



##########
#Read robot name and path#
#########
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gripper-name",
    type=str,
    default="Cobotta_Pro_900",
    help="Key to use to access RMPflow config files for a specific robot.",
)
parser.add_argument(
    "--usd-path",
    type=str,
    default="/Isaac/Robots/Denso/cobotta_pro_900.usd",
    help="Path to supported robot on Nucleus Server",
)
args = parser.parse_args()

########
#World genration
########
my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

#######
#Add primitive shape
#####
cube_prim_path = find_unique_string_name(
    initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
)
cube = scene.add(
    DynamicCuboid(
        name='cube',
        position=np.array([0.9, 0, 0.025]) / get_stage_units(),
        orientation=np.array([1, 0, 0, 0]),
        prim_path=cube_prim_path,
        scale=np.array([0.05, 0.05, 0.05]) / get_stage_units(),
        size=1.0,
        color=np.array([255, 0, 0]),
    )
)



##########
#Add robot
##########
# robot_name = args.robot_name
# usd_path = get_assets_root_path() + args.usd_path
# prim_path = "/my_robot"

# add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
# robot = my_world.scene.add(Robot(prim_path=prim_path, name=robot_name))

# ur5_prim_path = find_unique_string_name(
# initial_name="/World/UR5", is_unique_fn=lambda x: not is_prim_path_valid(x)
# )
# ur5_robot_name = find_unique_string_name(
#     initial_name="my_ur5", is_unique_fn=lambda x: not scene.object_exists(x)
# )
# ur5_robot = UR5(prim_path=ur5_prim_path, name=ur5_robot_name, attach_gripper=False, gripper_usd=args.gripper_name)
# ur5_robot.set_joints_default_state(
#     positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
# )

# assets_root_path = get_assets_root_path()
# if assets_root_path is None:
#     raise Exception("Could not find Isaac Sim assets folder")
# asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
# add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur5")

# gripper_usd = assets_root_path + "/Isaac/Robots/Robotiq/2F-140/2f140_instanceable.usd"
# # gripper_usd = assets_root_path + "/Isaac/Robots/Robotiq/2F-140/Props/instanceable_meshes.usd"
# add_reference_to_stage(usd_path=gripper_usd, prim_path="/World/ur5/tool0")

# gripper = ParallelGripper(
#     # We chose the following values while inspecting the articulation
#     end_effector_prim_path="/World/ur5/tool0",
#     # joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
#     joint_prim_names=["shoulder_pan_joint", "shoulder_lift_joint"],
#     joint_opened_positions=np.array([0.0, 0.0]),
#     joint_closed_positions=np.array([0.628, -0.628]),
#     action_deltas=np.array([0.05, 0.05]) / get_stage_units(),
# )
# # define the manipulator
# my_ur5 = my_world.scene.add(
#     SingleManipulator(
#         prim_path="/World/ur5",
#         name="my_ur5",
#         end_effector_prim_name="tool0",
#         gripper=gripper,
#     )
# )
# my_ur5.set_joints_default_state(
#     positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
# )

# assets_root_path = get_assets_root_path()
# if assets_root_path is None:
#     raise Exception("Could not find Isaac Sim assets folder")
# asset_path = "workspace/ur5_robotiq_2f140.usd"
# add_reference_to_stage(usd_path=asset_path, prim_path="/World")
# gripper = ParallelGripper(
#     end_effector_prim_path="/World/ur5/tool0",
#     joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
#     joint_opened_positions=np.array([0, 0]),
#     joint_closed_positions=np.array([0.628, -0.628]),
#     action_deltas=np.array([-0.2, 0.2]),
# )
# my_ur5 = SingleManipulator(
#     prim_path="/World/ur5",
#     name="my_ur5",
#     end_effector_prim_name="tool0",
#     gripper=gripper,
# )

# my_ur5.set_joints_default_state(
#     positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
# )

# my_world.reset()

franka_prim_path = find_unique_string_name(
    initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
)
franka_robot_name = find_unique_string_name(
    initial_name="my_franka", is_unique_fn=lambda x: not scene.object_exists(x)
)
my_franka = Franka(prim_path=franka_prim_path, name=franka_robot_name)

hand_camera = Camera(
    prim_path="/World/Franka/panda_hand/geometry/realsense/realsense_camera",
    frequency=20,
    resolution=(256, 256),
)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([1.0, 1.0, 1.0]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([-90, 45, 0]), degrees=True),
)

my_world.reset()
hand_camera.initialize()
camera.initialize()


i = 0
# camera.add_motion_vectors_to_frame()
hand_camera.add_distance_to_camera_to_frame()
hand_camera.add_instance_segmentation_to_frame()
hand_camera.add_instance_id_segmentation_to_frame()
hand_camera.add_semantic_segmentation_to_frame()
# hand_camera.set_focal_length(0.001)
hand_camera.set_focus_distance(4)
hand_camera.set_horizontal_aperture(0.05)
# hand_camera.set_vertical_aperture(0.05)

camera.add_distance_to_camera_to_frame()
camera.add_instance_segmentation_to_frame()
camera.add_instance_id_segmentation_to_frame()
camera.add_semantic_segmentation_to_frame()
hand_camera.set_focal_length(0.001)
# camera.set_focus_distance(4)
camera.set_horizontal_aperture(15)
# hand_camera.set_vertical_aperture(15)

stage = get_current_stage()
# plane_prim = stage.DefinePrim("World/defaultGroundPlane")
cube_prim = stage.DefinePrim("/World/Cube")
# add_update_semantics(prim=plane_prim, semantic_label="plane")
add_update_semantics(prim=cube_prim, semantic_label="cube")

# add_update_semantics(prim="/World/Cube", semantic_label="cube", type_label="class")

my_world.reset()

while simulation_app.is_running():
    my_world.step(render=True)
    
    if i == 500:
        # points_2d = camera.get_image_coords_from_world_points(
        #     np.array([cube.get_world_pose()[0], my_franka.get_world_pose()[0]])
        # )
        # points_3d = camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
        
        hand_rgb_image = hand_camera.get_rgba()[:, :, :3]
        hand_depth_image = hand_camera.get_current_frame()["distance_to_camera"]
        hand_instance_segmentation_image = hand_camera.get_current_frame()["instance_segmentation"]["data"]
        hand_instance_id_segmentation_image = hand_camera.get_current_frame()["instance_id_segmentation"]["data"]
        hand_semantic_segmentation_image = hand_camera.get_current_frame()["semantic_segmentation"]["data"]
        # focal_length = hand_camera.get_focal_length()
        focus_distance = hand_camera.get_focus_distance()
        horizontal_aperture = hand_camera.get_horizontal_aperture()
        # vertical_aperture = hand_camera.get_vertical_aperture()
        # nominal_width, nominal_height, optical_centre_x, optical_centre_y, max_fov, polynomial = hand_camera.get_fisheye_polynomial_properties()
        # print(focal_length)
        # print(focus_distance)
        # print(horizontal_aperture)
        # print(vertical_aperture)
        # print(max_fov)
        # print(hand_camera.get_current_frame()["instance_id_segmentation"])
        print(hand_camera.get_current_frame()["instance_segmentation"])
        print(np.unique(hand_instance_segmentation_image))
        print(hand_camera.get_current_frame()["semantic_segmentation"])
        print(np.unique(hand_semantic_segmentation_image))
        
        rgb_image = camera.get_rgba()[:, :, :3]
        depth_image = camera.get_current_frame()["distance_to_camera"]
        instance_segmentation_image = hand_camera.get_current_frame()["instance_segmentation"]["data"]
        instance_id_segmentation_image = hand_camera.get_current_frame()["instance_id_segmentation"]["data"]
        semantic_segmentation_image = hand_camera.get_current_frame()["semantic_segmentation"]["data"]
        
        hand_imgplot = plt.imshow(hand_rgb_image)
        plt.show()
        hand_depthplot = plt.imshow(hand_depth_image)
        plt.show()
        hand_inssegplot = plt.imshow(hand_instance_segmentation_image)
        plt.show()
        hand_insidsegplot = plt.imshow(hand_instance_id_segmentation_image)
        plt.show()
        hand_semsegplot = plt.imshow(hand_semantic_segmentation_image)
        plt.show()
        
        # hand_imgplot = plt.imshow(rgb_image)
        # plt.show()
        # hand_depthplot = plt.imshow(depth_image)
        # plt.show()
        # hand_inssegplot = plt.imshow(instance_segmentation_image)
        # plt.show()
        # hand_insidsegplot = plt.imshow(instance_id_segmentation_image)
        # plt.show()
        # hand_semsegplot = plt.imshow(semantic_segmentation_image)
        # plt.show()
        
        
    i += 1