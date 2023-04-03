from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.core.utils.stage import get_current_stage, get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.extensions import get_extension_path_from_name
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
import omni.kit.asset_converter
import omni.kit.commands
from pxr import Sdf, Gf, UsdPhysics, UsdLux, PhysxSchema
from omni.isaac.core.materials import OmniPBR, OmniGlass
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.urdf import _urdf
from omni.kit.importer.cad import _cad_importer

import numpy as np
import argparse
import matplotlib.pyplot as plt
import PIL
import cv2
import os
import random


########
#World genration
########
my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

#######
#Add primitive shape
#####
# cube_prim_path = find_unique_string_name(
#     initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
# )
# cube = scene.add(
#     DynamicCuboid(
#         name='cube',
#         position=np.array([0.9, 0, 0.025]) / get_stage_units(),
#         orientation=np.array([1, 0, 0, 0]),
#         prim_path=cube_prim_path,
#         scale=np.array([0.05, 0.05, 0.05]) / get_stage_units(),
#         size=1.0,
#         color=np.array([255, 0, 0]),
#     )
# )

#######
#Add ycb object
#########

# Setting up import configuration:
# status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
# urdf_interface = _urdf.acquire_urdf_interface()
import_config = _urdf.ImportConfig()
# import_config = _cad_importer.ImportConfig()

import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.make_default_prim = True
import_config.self_collision = False
import_config.create_physics_scene = False
# import_config.default_drive_strength = 1047.19751
# import_config.default_position_drive_damping = 52.35988
# import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.distance_scale = 0.5

path = '/home/nam/workspace/preprocessed_ycb'
objects = os.listdir(path)
obs = []
for i in range(3):
    a = random.randint(0, len(objects)-1)
    print(objects[a])
    # ob_prim_path = find_unique_string_name(
    #     initial_name="/World/{}".format(objects[a]), is_unique_fn=lambda x: not is_prim_path_valid(x)
    # )
    ycb_file = os.path.join(path, objects[a], 'raw_mesh.urdf')
    # Import URDF, stage_path contains the path the path to the usd prim in the stage.
    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path = ycb_file,
        import_config = import_config,
        # name = "/World/{}".format(objects[a]),
        # position=Gf.Vec3f(0, 0, -50),
        # color=Gf.Vec3f(0.5),
    )
    ob = XFormPrim(stage_path)
    
    ob_local_scale = ob.get_local_scale()
    ob_world_scale = ob.get_world_scale()
    ob_local_pose = ob.get_local_pose()
    ob_world_pose = ob.get_world_pose()
    print(ob_local_scale)
    print(ob_world_scale)
    print(ob_local_pose)
    print(ob_world_pose)
    
    ob.set_world_pose(np.array([1, 0.3, ob_world_scale[2]/2]), np.array([1, 0, 0, 0]))
    print(ob.get_local_pose())
    print(ob.get_world_pose())

    print(status, stage_path)
    
    textured_material = OmniPBR(prim_path='{}/omniPBR'.format(stage_path), texture_path=os.path.join('/home/nam/workspace/YCB', objects[a], 'google_16k/texture_map.png'))
    ob.apply_visual_material(visual_material=textured_material, weaker_than_descendants=False)
    print(ob.is_visual_material_applied())
    # scene.add(ob)
    
    obs.append(ob)

my_world.reset()

while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.current_time_step_index == 30:
        print('ob1: {}'.format(obs[0].get_world_pose()))