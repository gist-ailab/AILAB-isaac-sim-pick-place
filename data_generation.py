from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.universal_robots import UR10
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.sensor import Camera
from ur5 import UR5
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.syntheticdata import SyntheticData
import omni.replicator.core as rep
from pick_place_controller import PickPlaceController
import numpy as np
import argparse
import matplotlib.pyplot as plt
import PIL
from PIL import Image as im
import torchvision.transforms as T
import os
import random



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


ur5e_usd_path = "/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/github_my/isaac-sim-pick-place/ur5e_handeye_gripper.usd"
if os.path.isfile(ur5e_usd_path):
    pass
else:
    raise Exception(f"{ur5e_usd_path} not found")

add_reference_to_stage(usd_path=ur5e_usd_path, prim_path="/World/UR5")
gripper = ParallelGripper(
    end_effector_prim_path="/World/UR5/right_inner_finger_pad",
    joint_prim_names=["left_outer_knuckle_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0.0, 0.0]),
    joint_closed_positions=np.array([np.pi*2/9, -np.pi*2/9]),
    action_deltas=np.array([-np.pi*2/9, np.pi*2/9]),
)
my_ur5e = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/UR5", name="my_ur5e", end_effector_prim_name="right_inner_finger_pad", gripper=gripper
    )
)

hand_camera = Camera(
    prim_path="/World/UR5/realsense/RGB",
    frequency=20,
    resolution=(1920, 1080),
)

my_world.reset()
hand_camera.initialize()

my_controller = PickPlaceController(
    name="pick_place_controller", gripper=my_ur5e.gripper, robot_articulation=my_ur5e, end_effector_initial_height=0.5
)
articulation_controller = my_ur5e.get_articulation_controller()


i = 0
hand_camera.add_distance_to_camera_to_frame()
hand_camera.add_instance_segmentation_to_frame()

stage = get_current_stage()

my_world.reset()

transform = T.ToPILImage()


while simulation_app.is_running():
    my_world.step(render=True)
    
    # if i % 150 == 1:
    #     size = random.random()*0.09+0.03
    #     pos_y = random.random()*1.2+0.8
    #     pos_x = random.random()*1.2-0.4
    #     pos_z = size/2
        
    #     cube_prim_path = find_unique_string_name(
    #     initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x))   
    #     cube = scene.add(
    #         DynamicCuboid(
    #             name='cube',
    #             position=np.array([pos_x, pos_y, pos_z]),
    #             orientation=np.array([1, 0, 0, 0]),
    #             prim_path=cube_prim_path,
    #             scale=np.array([size, size, size]),
    #             size=1.0,
    #             color=np.array([255, 0, 0]),
    #         )
    #     )
    #     # plane_prim = stage.DefinePrim("World/defaultGroundPlane")
    #     cube_prim = stage.DefinePrim("/World/Cube")
    #     # add_update_semantics(prim=plane_prim, semantic_label="plane")
    #     add_update_semantics(prim=cube_prim, semantic_label="cube")
        
    #     my_world.reset()
        
    if i % 150 == 1:
        size_z = (random.random()*0.09+0.03)
        size = size_z/2
        pos_y = random.random()*1.2+0.8
        pos_x = random.random()*1.2-0.4
        pos_z = size_z/2
        
        cube_prim_path = find_unique_string_name(
        initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x))   
        cube = scene.add(
            DynamicCylinder(
                name='cube',
                position=np.array([pos_x, pos_y, pos_z]),
                orientation=np.array([1, 0, 0, 0]),
                prim_path=cube_prim_path,
                scale=np.array([size, size, size_z]),
                color=np.array([255, 0, 0]),
            )
        )
        cube_prim = stage.DefinePrim("/World/Cube")
        add_update_semantics(prim=cube_prim, semantic_label="cube")
        
        my_world.reset()
        
    if i % 150 == 149:
        hand_rgb_image = hand_camera.get_rgba()[:, :, :3]
        hand_depth_image = hand_camera.get_current_frame()["distance_to_camera"]
        hand_instance_segmentation_image = hand_camera.get_current_frame()["instance_segmentation"]["data"]
        focus_distance = hand_camera.get_focus_distance()
        horizontal_aperture = hand_camera.get_horizontal_aperture()
        
        print(hand_camera.get_current_frame()["instance_segmentation"])
        print(np.unique(hand_instance_segmentation_image))
        
        hand_imgplot = transform(hand_rgb_image)
        # hand_imgplot.save("/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/github_my/cylinder/img/img"+str(int(i/150))+".png")
       
        hand_inssegplot = im.fromarray(hand_instance_segmentation_image)
        # hand_inssegplot.save("/home/ailab/Workspace/minhwan/isaac_sim-2022.2.0/github_my/cylinder/mask/mask"+str(int(i/150))+".png")
    
        scene.remove_object('cube')
        my_world.reset()
    if i == 15000:
        simulation_app.close()

    i += 1
    