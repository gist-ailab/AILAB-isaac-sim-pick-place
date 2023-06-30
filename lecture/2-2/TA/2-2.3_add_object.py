# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.3 Basic simulation loop with primitive object load
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects import DynamicCuboid       #

import numpy as np                                      #


my_world = World(stage_units_in_meters=1.0)


scene = Scene()
scene.add_default_ground_plane()


max_ep_step = 15                                        #
ep_num = 0                                              #
max_ep_num = 100                                        #

obj_idx = 0                                             #

while simulation_app.is_running():
    scale = list(np.random.rand(3) * 0.5)               #
    position = [0, 0, scale[2]/2]                       #
    cube = DynamicCuboid("/World/object"+str(obj_idx), position=position, scale=scale)  #
    obj_idx += 1                                        #
    
    ep_num += 1                                         #
    print("Start Episode: ", ep_num)                    #
    
    for ep_step in range(max_ep_step):                  #
        my_world.step(render=True)
        print("Simulation Episode Step: ", ep_step)     #
    
    print("End Episode: ", ep_num)                      #
    
    if ep_num >= max_ep_num:                            #
        simulation_app.close()