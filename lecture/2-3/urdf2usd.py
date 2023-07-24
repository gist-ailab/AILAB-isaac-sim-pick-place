import numpy as np
import os
import sys

lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
sys.path.append(lecture_path)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.kit.asset_converter
import omni.kit.commands
from omni.isaac.core.prims.xform_prim import XFormPrim


# Setting up import configuration:
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

import_config.merge_fixed_joints = False
import_config.convex_decomp = False
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.make_default_prim = True
import_config.self_collision = False
import_config.create_physics_scene = False
import_config.distance_scale = 0.5

path = os.path.join(lecture_path, 'dataset/preprocessed_1')
save_path = os.path.join(lecture_path, 'dataset/test/')
objects = os.listdir(path)
obs = []

for i in range(len(objects)):
    print(objects[i])
    ycb_file = os.path.join(path, objects[i], 'raw_mesh.urdf')
    
    # Import URDF, stage_path contains the path the path to the usd prim in the stage.
    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path = ycb_file,
        import_config = import_config,
        dest_path = save_path+objects[i]+'/final.usd',
    )
    ob = XFormPrim(stage_path)
    
    ob_world_scale = ob.get_world_scale()

    ob.set_world_pose(np.array([1, 0.3, ob_world_scale[2]/2]), np.array([1, 0, 0, 0]))

    # textured_material = OmniPBR(prim_path='{}/Looks'.format(stage_path), texture_path=os.path.join('/home/ailab/Workspace/minhwan/YCB', objects[i], 'google_16k/texture_map.png'))
    print(ob.is_visual_material_applied())
   
    obs.append(ob)