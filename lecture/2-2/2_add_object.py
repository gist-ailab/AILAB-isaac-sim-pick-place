# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.3 Basic simulation loop with primitive object load
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects.cuboid import DynamicCuboid, FixedCuboid, VisualCuboid
# from omni.isaac.core.objects.sphere import DynamicSphere
# from omni.isaac.core.objects.cylinder import DynamicCylinder
# from omni.isaac.core.objects.cone import DynamicCone
# from omni.isaac.core.objects.capsule import DynamicCapsule

import numpy as np                                      #



"""
There are three types of cuboid: dynamic, fixed, visual

1. visual cuboid is only for visualization and does not interact with physics
2. fixed cuboid = visual cuboid + physics(collision) | static cuboid
3. dynamic cuboid = fixed cuboid + physics(dynamics)

"""

# World 초기화
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

my_world.initialize_physics()

# Main Loop
frame = 0
while simulation_app.is_running():
    print(f"Simulation Step: {frame}")
    my_world.step(render=True)

    # Add cuboid
    if frame == 10:
        cube = VisualCuboid(prim_path="/World/visual",
                            name="visual_cube",
                            position=[0.3, 0.3, 0.1], 
                            scale=[0.1, 0.1, 0.1],
                            color=np.array([1.0, 0.0, 0.0]))
        my_world.scene.add(cube)
    elif frame == 20:
        cube = FixedCuboid(prim_path="/World/fixed",
                           name="fixed_cube",
                           position=[0.5, 0.3, 0.1], 
                           scale=[0.1, 0.1, 0.1], 
                           color=np.array([0.0, 1.0, 0.0]))
        my_world.scene.add(cube)
    elif frame == 30:
        cube = DynamicCuboid(prim_path="/World/dynamic1", 
                             name='dynamic_cube1',
                             position=[0.3, 0.3, 0.3], 
                             scale=[0.1, 0.1, 0.1], 
                             color=np.array([0.0, 0.0, 1.0]))
        my_world.scene.add(cube)
    elif frame == 40:
        cube = DynamicCuboid(prim_path="/World/dynamic2", 
                             name='dynamic_cube2',
                             position=[0.5, 0.3, 0.3], 
                             scale=[0.1, 0.1, 0.1], 
                             color=np.array([0.0, 0.0, 1.0]))
        my_world.scene.add(cube)
    
    elif frame > 100:
        break
    frame += 1

simulation_app.close()
print("Simulation is Closed")

