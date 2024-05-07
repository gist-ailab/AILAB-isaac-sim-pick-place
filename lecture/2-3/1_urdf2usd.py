import numpy as np
import os
import sys

lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # path to lecture
sys.path.append(lecture_path)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})
sys.path.append('/isaac-sim/extscache/omni.kit.asset_converter-1.3.4+lx64.r.cp37')
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

path = os.path.join(lecture_path, '2-3/urdf')
save_path = os.path.join(lecture_path, '2-3/usd')
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
        dest_path = os.path.join(save_path,objects[i],'final.usd'),
    )
    
simulation_app.close()
