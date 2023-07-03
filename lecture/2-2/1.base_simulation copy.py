#----- import necessary packages -----#
# initialize simulation app before import other isacc packages, waiting for about 240 seconds when first run
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

#----- Initialize simulation setting -----#
# initialize simulation world
sim_step = 0
my_world = World(stage_units_in_meters=1.0)


# initialize simulation scene
scene = Scene()
scene.add_default_ground_plane() # add default ground plane

#----- Simulation loop -----#
sim_step = 0
max_sim_step = 10000
while simulation_app.is_running():
    my_world.step(render=True)
    print("Simulation Step: ", sim_step)
    # close simulation app
    if sim_step == max_sim_step:
        simulation_app.close()