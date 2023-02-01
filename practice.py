from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
from omni.isaac.universal_robots import UR10
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper

import numpy as np
import argparse

from ur5 import UR5

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
        position=np.array([3, 3, 0.025]) / get_stage_units(),
        orientation=np.array([1, 0, 0, 0]),
        prim_path=cube_prim_path,
        scale=np.array([0.05, 0.05, 0.05]) / get_stage_units(),
        size=1.0,
        color=np.array([0, 0, 1]),
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

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise Exception("Could not find Isaac Sim assets folder")
asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur5")

gripper_usd = assets_root_path + "/Isaac/Robots/Robotiq/2F-85/2f85_instanceable.usd"
add_reference_to_stage(usd_path=gripper_usd, prim_path="/World/ur5/tool0")
# define the gripper
gripper = ParallelGripper(
    # We chose the following values while inspecting the articulation
    end_effector_prim_path="/World/ur5/tool0",
    joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
    joint_opened_positions=np.array([0.0, 0.0]),
    joint_closed_positions=np.array([0.628, -0.628]),
    action_deltas=np.array([-0.628, 0.628]),
)
# define the manipulator
my_ur5 = my_world.scene.add(
    SingleManipulator(
        prim_path="/World/ur5",
        name="ur5",
        end_effector_prim_name="tool0",
        gripper=gripper,
    )
)

while simulation_app.is_running():
    my_world.step(render=True)