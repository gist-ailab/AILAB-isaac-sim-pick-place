import os
import sys
lecture_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__))) # path to lecture
sys.path.append(lecture_path)
print(lecture_path)
from os import listdir
from os.path import join, isfile
import trimesh
from natsort import natsorted
from tqdm import tqdm


from object2urdf import ObjectUrdfBuilder


def get_file_list(path):
    file_list = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return file_list


def get_YCB_mesh(data_root, texture=False):
    """get YCB raw mesh from data_root
    Args:
            data_root (str): [ycb dataset root]
            looks like >>>
            ycb-tools/models/ycb # data root
            ├── 001_chips_can
            ├── 002_master_chef_can
            ├── ...
            ├── 076_timer
            └── 077_rubiks_cube
            
            texture (bool): get ycb mesh with texture(*.dae)
            *but cannot convert to point cloud
    
    Returns:
            meta_info (dict): dict(object name, path to mesh file) for All google_16k mesh
            
    """
    data_info = {}
    ycb_list = [os.path.join(data_root, p) for p in os.listdir(data_root)]
    ycb_sorted_list = natsorted(ycb_list)
    for target_ycb_folder in tqdm(ycb_sorted_list):
        target_name = target_ycb_folder.split('/')[-1]
        target_name = target_name.replace("-", "_")
        
        # google_16k has high quality
        if "google_16k" in os.listdir(target_ycb_folder):
            pass
        else:
            continue

        if not texture:
            mesh_file = os.path.join(target_ycb_folder, "google_16k", "nontextured.ply")
        else:
            mesh_file = os.path.join(target_ycb_folder, "google_16k", "textured.obj")
        data_info[target_name] = mesh_file
    
    meta_info = {}
    meta_idx = 0
    for target_name, mesh_file in data_info.items():
        meta_info[target_name] = [meta_idx, mesh_file]
        meta_idx += 1
    
    return meta_info

def preprocess_ycb(data_root, save_root):
    os.makedirs(save_root, exist_ok=True)
    
    mesh_info = get_YCB_mesh(data_root, texture=True)

    for obj_name, v in mesh_info.items(): 
        obj_dir = os.path.join(save_root, obj_name)
        os.makedirs(obj_dir, exist_ok=True)

        #raw_mesh 
        mesh_file = v[1]
        mesh = trimesh.load_mesh(mesh_file)
        mesh.export(os.path.join(obj_dir, "raw_mesh.obj"))

        #for urdf
        root = os.path.join(obj_dir, "_prototype.urdf")
        build_prototype_urdf(root)

        obj2urdf(root=obj_dir)


    for obj_name, _ in mesh_info.items():
        obj_dir = os.path.join(save_root, obj_name)
        origin = obj_dir + "/" + "raw_mesh.obj.urdf"
        target = obj_dir + "/" + "raw_mesh.urdf"
        files = get_file_list(obj_dir)

        source = [x for x in files if x in origin]
        if len(source) == 0 or len(source) == 1:
            continue
        else:
            os.rename(source[1], target)
    

def build_prototype_urdf(root):

    output_name = root
    with open(output_name, "w") as f:
        text = """<?xml version="1.0" ?>
    <robot name="_name">
        <link name="_name">
            <visual>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <geometry>
                    <mesh filename="_prototype.obj" scale="1 1 1" />
                </geometry>
                <material name="texture">
                    <color rgba="1.0 1.0 1.0 1.0"/>
                </material>
            </visual>
            <collision>
                <geometry>
                    <mesh filename="_prototype.obj" scale="1 1 1" />
                </geometry>
            </collision>
            <inertial>
                <mass value="1.0"/>
                <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
            </inertial>
        </link>
    </robot>
        """
        f.write(text)
        f.close()


def obj2urdf(root):
    builder = ObjectUrdfBuilder(root)
    builder.build_library(force_overwrite=True, decompose_concave=True, force_decompose=True, center='top')


if __name__ == "__main__":
    data_root = os.path.join(lecture_path, "dataset/origin_YCB")
    save_root = os.path.join(lecture_path, "2-3/urdf")
    if not os.path.isdir(save_root):    
        os.mkdir(save_root)

    preprocess_ycb(data_root=data_root, save_root=save_root)