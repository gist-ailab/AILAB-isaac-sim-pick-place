import os
import trimesh
import sys

lecture_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))) # path to lecture
sys.path.append(lecture_path)
from file_utils import *

from preprocess_utils import DataProcessing

from object2urdf import ObjectUrdfBuilder

from pathlib import Path
sys.path.append(str(Path().absolute()))


def preprocess_ycb(data_root, save_root):
    os.makedirs(save_root, exist_ok=True)
    
    mesh_info = DataProcessing.get_YCB_mesh(data_root, texture=True)

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
    save_root = os.path.join(lecture_path, "dataset/test")
    if not os.path.isdir(save_path):    
        os.mkdir(save_path)

    preprocess_ycb(data_root=data_root, save_root=save_root)