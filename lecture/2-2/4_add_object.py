# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.4 Basic simulation loop with primitive object load
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCone       #

import numpy as np                                      #

my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

scale = list(np.random.rand(3) /2)               
position = [0, 0, scale[2]/2]                       
cube = DynamicCuboid(
    prim_path="/World/object", 
    position=position, 
    scale=scale)

while simulation_app.is_running():
    my_world.step(render=True)