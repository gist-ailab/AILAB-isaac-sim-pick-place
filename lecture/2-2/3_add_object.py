# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.4 Basic simulation loop with primitive object load
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects.cuboid import DynamicCuboid, FixedCuboid, VisualCuboid
from omni.isaac.core.objects.sphere import DynamicSphere
from omni.isaac.core.objects.cylinder import DynamicCylinder
from omni.isaac.core.objects.cone import DynamicCone
from omni.isaac.core.objects.capsule import DynamicCapsule

import numpy as np                                      #

my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

scale = list(np.random.rand(3)/2)               
position = [1, 0, scale[2]/2]                       
cube = DynamicCuboid(                        # DynamicCuboid 선언 및 생성
    prim_path="/World/dynamic", 
    position=position,
    orientation=[0.5,0.5,0.5,0.5], 
    scale=scale)

scale = list(np.random.rand(3)/2)               
position = [0, 0, scale[2]/2]                       
cube = FixedCuboid(                          # FixedCuboid 선언 및 생성
    prim_path="/World/fixed", 
    position=position,
    orientation=[0.5,0.5,0.5,0.5], 
    scale=scale)

scale = list(np.random.rand(3)/2)               
position = [-1, 0, scale[2]/2]                       
cube = VisualCuboid(                          # VisualCuboid 선언 및 생성
    prim_path="/World/visual",  
    position=position,
    orientation=[0.5,0.5,0.5,0.5], 
    scale=scale)

while simulation_app.is_running():
    my_world.step(render=True)