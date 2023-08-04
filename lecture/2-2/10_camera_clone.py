# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-2.5 Basic simulation loop with camera (RGB)
# ---- ---- ---- ----

# 수업 중 말씀드렸던 cloner에 대한 코드입니다. 
# 해당 코드는 xformprim X를 선언하여 카메라와 object를 X라는 prim의 하위 prim으로 선언하였습니다. 
# grid cloner를 통해 4개로 clone하여 총 5개의 이미지를 동시에 얻을 수 있습니다.
# 하지만, 1개의 카메라를 사용했을 때는 10 step 정도의 안정화 스텝이 필요한 반면 저희가 구현한 해당 코드에서는 200~300 정도의 step이 필요합니다.
# 이미지도 기존 (1920,1080)의 해상도는 메모리 부족으로 가능하지 않아 (960,540)으로 지정하였습니다.
# 이미지를 획득하는 과정에서는 오히려 cloner를 쓰는 것이 data 수집 성능이 저하된다고 말할 수 있습니다.


import os 
import sys
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene

from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCone       #

from omni.isaac.core.prims.xform_prim import XFormPrim

from omni.isaac.cloner import GridCloner
import omni.replicator.core as rep
from omni.isaac.sensor import Camera                            
                                      
from PIL import Image  
import numpy as np                                      

my_world = World(stage_units_in_meters=1.0)

scene = Scene()
scene.add_default_ground_plane()

x = XFormPrim("/World/X")                           # object와 camera의 상위 prim x 선언

scale = list(np.random.rand(3) * 0.2)               
position = [0.3, 0.3, scale[2]/2]                       
cube = DynamicCuboid(
    prim_path="/World/X/object", 
    position=position, 
    scale=scale)

my_world.reset()

save_root = os.path.join(lecture_path, "2-2/sample_data")            
print("Save root: ", save_root)                                 
os.makedirs(save_root, exist_ok=True)                           

def save_image(image, path):                                    
    image = Image.fromarray(image)                              
    image.save(path)                                            


my_camera = Camera(                                             
    prim_path="/World/X/RGB",                                     
    frequency=20,                                               
    resolution=(960, 540),                                    
    position=[0.48176, 0.13541, 0.71],             # 로봇 팔의 위치   
    orientation=[0.5,-0.5,0.5,0.5]                 # quaternion                 
)                                                               
my_camera.set_focal_length(1.93)                   # 카메라 설정값 세팅      
my_camera.set_focus_distance(4)                                 
my_camera.set_horizontal_aperture(2.65)                        
my_camera.set_vertical_aperture(1.48)                           
my_camera.set_clipping_range(0.01, 10000)                       

my_camera.initialize()                             # 카메라 시작                

cloner = GridCloner(spacing=3)                      # cloner 선언
target_paths = cloner.generate_paths("/World/X",4)
cloner.clone("/World/X", target_paths)

ep_num = 0
max_ep_num = 10

resol = (960, 540)
rgb_annotators = []

render_product = rep.create.render_product("World/X/RGB", resol)                # clone된 카메라의 데이터에 접근하기 위한 replica 함수 사용
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])
rgb_annotators.append(rgb)
for target_path in target_paths:
    render_product = rep.create.render_product(target_path + "/RGB", resol)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    rgb_annotators.append(rgb)

rep.orchestrator.run()

while simulation_app.is_running():
    ep_num += 1                                      
    my_world.step(render=True)
    print("Episode: ", ep_num)
                   
    if ep_num == max_ep_num:                       # 300 step 이후
        for i, rgb_annotator in enumerate(rgb_annotators):
            rgb_image = rgb_annotator.get_data()
            save_image(rgb_image, os.path.join(save_root, "rgb_{}.png".format(i)))  # 사진 저장
        simulation_app.close()
simulation_app.close()