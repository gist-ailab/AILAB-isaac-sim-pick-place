# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.3 Basic simulation loop with primitive object load
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

sim_step = 0
max_sim_step = 10000
while simulation_app.is_running():
    my_world.step(render=True)

    scale = list(np.random.rand(3) * 0.5)               
    position = [0, 0, scale[2]/2]                       
    cube = DynamicCuboid(
        prim_path="/World/object", 
        position=position, 
        scale=scale)
    
    sim_step += 1
    if sim_step >= max_sim_step:
        simulation_app.close()
