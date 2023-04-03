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

path = '/home/nam/workspace/YCB/002_master_chef_can/google_16k/textured.obj'
_si = _cad_importer.acquire_interface()
part = _cad_importer.Part()
stepfile = _si.load_cad_file(path)


my_world.reset()

while simulation_app.is_running():
    my_world.step(render=True)
    