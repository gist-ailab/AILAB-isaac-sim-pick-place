# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.2 Basic simulation loop with plane Scene
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})
print(simulation_app.DEFAULT_LAUNCHER_CONFIG)

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene          #


my_world = World(stage_units_in_meters=1.0)


scene = Scene()                                         #
scene.add_default_ground_plane()                        #


sim_step = 0
max_sim_step = 10000
while simulation_app.is_running():
    my_world.step(render=True)
    print("Simulation Step: ", sim_step)
    
    sim_step += 1
    if sim_step >= max_sim_step:
        simulation_app.close()