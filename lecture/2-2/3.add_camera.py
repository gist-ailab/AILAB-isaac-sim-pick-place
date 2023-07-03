#----- import necessary packages -----#
# initialize simulation app before import other isacc packages, waiting for about 240 seconds when first run
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) 
from omni.isaac.core import World
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.sensor import Camera

#----- save image -----#
import os
from PIL import Image
save_root = os.path.join(os.getcwd(), "sample_data")
print("Save root: ", save_root)
os.makedirs(save_root, exist_ok=True)
def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)


#----- Initialize simulation setting -----#
# initialize simulation world
sim_step = 0
my_world = World(stage_units_in_meters=1.0)

# initialize simulation scene
scene = Scene()
scene.add_default_ground_plane() # add default ground plane

# add camera to scene
my_camera = Camera(
    prim_path="/World/RGB",
    frequency=20,
    resolution=(1920, 1080), # 1920*1080
    position=[0.48176, 0.13541, 0.71], # attach camera to robot hand
    orientation=[0.5,-0.5,0.5,0.5] # quaternion 
)
my_camera.set_focal_length(1.93)
my_camera.set_focus_distance(4)
my_camera.set_horizontal_aperture(2.65)
my_camera.set_vertical_aperture(1.48)
my_camera.set_clipping_range(0.01, 10000)

my_camera.initialize()

#----- Simulation loop -----#
ep_num = 0
max_ep_num = 100
max_ep_step = 15

while simulation_app.is_running():
    # initialize episode
    ep_num += 1
    my_world.reset()
    print("Start Episode: ", ep_num)
    
    for ep_step in range(max_ep_step):
        # step episode
        my_world.step(render=True)
        print("Episode Step: ", ep_step)
        rgb_image = my_camera.get_rgba()
        save_image(rgb_image, os.path.join(save_root, "rgb_{:04d}_{:04d}.png".format(ep_num, ep_step)))
    
    # close simulation app
    if ep_num == max_ep_num:
        simulation_app.close()
