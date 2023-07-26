# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.3 Basic simulation loop with primitive object load
# ---- ---- ---- ----


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects import DynamicCuboid       

from omni.isaac.core.prims.xform_prim import XFormPrim

from omni.isaac.core.utils.prims import delete_prim
import numpy as np                   

my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

obj_idx = 0                                             

x = XFormPrim("/World/x", position=[10,0,0])            # World에 속하는 prim x,y,z 선언
y = XFormPrim("/World/y", position=[0,10,0])            
z = XFormPrim("/World/z", position=[0,0,10])

while simulation_app.is_running():
    scale = list(np.random.rand(3) * 0.5)               # 임의의 scale 선언
    position = [0, 0, scale[2]/2]                       # 0,0 으로 position 고정. scale[2]/2는 물체가 바닥에 놓여지게 하기 위함 
    my_world.step(render=True)
    
    if obj_idx % 600 == 0:
        cube = DynamicCuboid("/World/x/object"+str(int(obj_idx/100)), position=position, scale=scale)  # World/x에 object 생성
    if obj_idx % 600 == 100:
        delete_prim("/World/x/object"+str(int(obj_idx/100)-1))                                         # 해당 object 삭제
    if obj_idx % 600 == 200:
        cube = DynamicCuboid("/World/y/object"+str(int(obj_idx/100)), position=position, scale=scale)  # World/y에 object 생성
    if obj_idx % 600 == 300:
        delete_prim("/World/y/object"+str(int(obj_idx/100)-1))
    if obj_idx % 600 == 400:
        cube = DynamicCuboid("/World/z/object"+str(int(obj_idx/100)), position=position, scale=scale)  # World/z에 object 생성
    if obj_idx % 600 == 500:
        delete_prim("/World/z/object"+str(int(obj_idx/100)-1))
    
    obj_idx += 1 
    if obj_idx >= 60000:                            #
        simulation_app.close()