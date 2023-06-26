#----- import necessary packages -----#
# initialize simulation app before import other isacc packages, waiting for about 240 seconds when first run
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid

import numpy as np

#----- Initialize simulation setting -----#
# initialize simulation world
sim_step = 0
sim_world = World(stage_units_in_meters=1.0)

# initialize simulation scene
scene = Scene()
scene.add_default_ground_plane() # add default ground plane


#----- Simulation loop -----#
ep_num = 0
max_ep_num = 100
max_ep_step = 15
obj_idx = 0

while simulation_app.is_running():
    #----- initialize episode -----#
    # Add random cuboid to scene
    scale = list(np.random.rand(3) * 0.5) # random scale in [0, 0.5]
    position = [0, 0, scale[2]/2]
    cube = DynamicCuboid("/World/object"+str(obj_idx), position=position, scale=scale)
    obj_idx+=1
    
    ep_num += 1
    print("Start Episode: ", ep_num)
    
    for sim_step in range(max_ep_step):
        
        
        
        sim_world.step(render=True)
        print("Simulation Step: ", sim_step)
    
    print("End Episode: ", ep_num)
    
    # close simulation app
    if sim_step == max_ep_num:
        simulation_app.close()