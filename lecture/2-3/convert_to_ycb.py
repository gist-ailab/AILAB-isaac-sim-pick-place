from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
import omni.kit.asset_converter
import omni.kit.commands
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.prims.xform_prim import XFormPrim

import numpy as np
import os

########
#World genration
########
my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()


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

path = '/home/ailab/Downloads/preprocessed_ycb'
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
        dest_path = '/home/ailab/Workspace/minhwan/ycb/'+objects[i]+'/final.usd',
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
    
    textured_material = OmniPBR(prim_path='{}/Looks'.format(stage_path), texture_path=os.path.join('/home/ailab/Workspace/minhwan/YCB', objects[i], 'google_16k/texture_map.png'))
    print(ob.is_visual_material_applied())
   
    obs.append(ob)

my_world.reset()

while simulation_app.is_running():
    my_world.step(render=True)
    
    if my_world.current_time_step_index == 30:
        print('ob1: {}'.format(obs[0].get_world_pose()))