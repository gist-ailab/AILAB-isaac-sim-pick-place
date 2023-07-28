# ---- ---- ---- ----
# GIST-AILAB, 2023 summer school
# Day2. 
# 2-3.2 Data generation with YCB dataset 
# ---- ---- ---- ----

#-----0. preliminary -----#

# python path setting
import os
import sys
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)

# import packages
from PIL import Image
import numpy as np
import random

# function for save image
def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)

if __name__ == "__main__":
    
    # data and save path
    print(lecture_path)
    robot_path = os.path.join(lecture_path, 'utils/tasks/ur5e_handeye_gripper.usd')
    data_path = os.path.join(lecture_path, 'dataset/ycb')
    save_path = os.path.join(lecture_path, 'dataset/detect_img')
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'train/img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'val/img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test/img'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'train/mask'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'val/mask'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'test/mask'), exist_ok=True)
    
    # get object list
    obj_dirs = [os.path.join(data_path, obj_name) for obj_name in os.listdir(data_path)]
    obj_dirs.sort()
    print(obj_dirs)
    object_info = {}
    total_object_num = len(obj_dirs)
    for obj_idx, obj_dir in enumerate(obj_dirs):
        usd_file = os.path.join(obj_dir, 'final.usd')
        object_info[obj_idx] = {
            'name': os.path.basename(obj_dir),
            'usd_file': usd_file,
            'label': obj_idx+2, # set object label 2 ~ 
        }
    print(object_info)
    
    #-----1. Initialize simulation app and import packages
    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from omni.isaac.core import World
    from omni.isaac.core.utils.semantics import add_update_semantics
    from omni.isaac.core.utils.stage import get_current_stage
    from omni.isaac.core.utils.prims import create_prim, delete_prim

    my_world = World(stage_units_in_meters=1.0)

    #-----2. Define Task and add to the world
    """
    UR5ePickPlace
    - custom task for pick and place with UR5e + attached camera
    - TODO: summary of the task detail
    """
    from utils.tasks.pick_place_vision_task import UR5ePickPlace
    # add task to the world
    my_task = UR5ePickPlace(objects_list = [])  # releasing offset at the target position
    my_world.add_task(my_task)
    my_world.reset()

    # get robot, camera, stage from task
    task_params = my_task.get_params()

    my_ur5 = my_world.scene.get_object(task_params["robot_name"]["value"])
    hand_camera = my_task.get_camera()

    stage = get_current_stage()

    #-----4. Simulation loop
    if os.path.isfile(os.path.join(save_path, "train/img/img2.png")):
        i = 8219
    else:
        i = 0
        
    while simulation_app.is_running():
        
        my_world.step(render=True)

        #-----4.1. Randomly select object and add to the scene
        obj_num = random.randint(1,3) # random number of objects
        for l in range(obj_num):
            # randomize x,y position related to world, z position is determined by object size
            random_position = [random.random()*0.3+0.33, random.random()*0.6-0.17, 0.1]

            # random object from obj_dirs
            random_idx = random.randint(0, total_object_num-1)
            usd_file = object_info[random_idx]['usd_file']
            obj_label = object_info[random_idx]['label']
            prim_path = "/World/object"+str(l) # 1st, 2nd, 3rd object
            object_prim = create_prim(
                usd_path = usd_file, 
                prim_path = prim_path, 
                position = random_position, 
                scale = [0.2,0.2,0.2])
            # update semantic information
            # label 0   : unlabel 
            # label 1   : background 
            # label 2 ~ : YCB object 
            add_update_semantics(prim=object_prim, semantic_label=str(obj_label+100*l))
        
        # about 10 steps to stablize the scene after reset
        my_world.reset()
        for j in range(10):
            my_world.step(render=True)

        #-----4.2. Get camera image
        hand_rgb_image = hand_camera.get_rgba()[:, :, :3]
        current_frame = hand_camera.get_current_frame()
        hand_instance_segmentation_image = current_frame["instance_segmentation"]["data"]
        hand_instance_segmentation_dict = current_frame["instance_segmentation"]["info"]["idToSemantics"]
        
        print(hand_camera.get_current_frame()["instance_segmentation"])
        
        #-----4.3. post-processing for instance segmentation
        # class가 2,3,4로 순서대로 나타나는게 아니라 (2,3) (3,4) 등으로 나타날 때도 있음 해당 예외 처리를 위해 다음과 같은 dict 생성 
        class_dict = {}
        for k in range(2,5):
            if str(k) in hand_instance_segmentation_dict.keys():
                class_dict[k]=int(hand_instance_segmentation_dict[str(k)]['class'])
        
        # hand_instance_segmentation_image의 경우 class(2,3,4)로 라벨이 되어있음. 이를 label로 바꿔줌
        for c in class_dict.keys():
            np.place(hand_instance_segmentation_image, hand_instance_segmentation_image==c, class_dict[c])
        print(np.unique(hand_instance_segmentation_image))

        #-----4.4. save image
        # png형태로 저장
        if i%10==0:
            split = "val"
        elif i%10==1:
            split= "test"
        else:
            split = "train"
        img_name = "{}/img/img{}.png".format(split, i)
        mask_name = "{}/mask/mask{}.png".format(split, i)
        save_image(hand_rgb_image, os.path.join(save_path, img_name))
        save_image(hand_instance_segmentation_image, os.path.join(save_path, mask_name))
            
        # #-----4.5. delete objects
        for l in range(obj_num):
            delete_prim("/World/object"+str(l))
        my_world.reset()
        i += 1

        if i==10000:
            simulation_app.close()