# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.tasks as tasks
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import create_prim, get_prim_path
from omni.isaac.core.utils.stage import get_stage_units
from robots.ur5e_handeye import UR5eHandeye
import os, random
import numpy as np
from typing import Optional
from pxr import Gf
import omni.usd


class UR5ePickPlace(tasks.PickPlace):
    """[summary]

        Args:
            name (str, optional): [description]. Defaults to "ur5_pick_place".
            cube_initial_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_initial_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            target_position (Optional[np.ndarray], optional): [description]. Defaults to None.
            cube_size (Optional[np.ndarray], optional): [description]. Defaults to None.
            offset (Optional[np.ndarray], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str = "ur5e_pick_place",
        imported_list: Optional[list] = None,    # import mesh file such as stl, obj, etc.
        target_position: Optional[np.ndarray] = None,
        offest: Optional[np.ndarray] = np.array([0, 0, 0.15]),
    ) -> None:
        tasks.PickPlace.__init__(self, name=name, )
        self.pos_x = (random.random()*1.2-0.4)/4
        self.pos_y = (random.random()*1.2+0.8)/4
        self.pos_z = 0.1
        self.imported_objects = imported_list
        self.imported_objects_prim_path = "/World/object"
        
        # 아래 3개 변수는 개수가 많아진다면 list나 dict로 해야 함.
        self.position = None
        self.orientation = None
        self.task_object_name = None

        if self.imported_objects is None:
            cube_prim_path = "/World/Cube"
            cube_name = "cube"
            size_scale = 0.03
            self._object = DynamicCuboid(
                prim_path = cube_prim_path,
                name = cube_name,
                position = np.array([0.4, 0.33, 0.1 + size_scale/2]),
                scale = np.array([size_scale, size_scale, size_scale]),
                color = np.array([0, 0, 1]),
                size = 1.0,
                mass = 0.01,
            )
            self._objects = self._object
        else:
            imported_object = create_prim(
                                          usd_path=imported_list[random.randint(0, len(imported_list)-1)],
                                        #   usd_path=imported_list[0],
                                          prim_path=self.imported_objects_prim_path,
                                          position=[self.pos_x, self.pos_y, self.pos_z],
                                          scale=[0.2,0.2,0.2])
            # self._objects = imported_object
            self._objects = imported_object
            ''' 위 부분은 for문을 통해 여러개의 object를 추가할 수 있도록 수정해야함 '''

        self._target_position = target_position
        self._offset = offest
        if self._target_position is None:
            self._target_position = np.array([0.4, -0.33, 0])
            self._target_position[2] = 0.05 # considering the length of the gripper tip
        self._target_position = self._target_position + self._offset
        return



    def set_up_scene(self, scene: Scene) -> None:
        """[summary]
        YCB objects are added to the scene. If the ycb objects are not found in the scene, 
        only the cuboid added to the scene.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        scene.add_default_ground_plane()

        if self.imported_objects is None:
            self._task_object = scene.add(self._objects)

        self._robot = self.set_robot()
        scene.add(self._robot)

        self._move_task_objects_to_their_frame()
        return


    def set_robot(self) -> UR5eHandeye:
        """[summary]

        Returns:
            UR5e: [description]
        """
        working_dir = os.path.dirname(os.path.realpath(__file__))
        ur5e_usd_path = os.path.join(working_dir, "ur5e_handeye_gripper.usd")
        if os.path.isfile(ur5e_usd_path):
            pass
        else:
            raise Exception(f"{ur5e_usd_path} not found")
        
        ur5e_prim_path = find_unique_string_name(
            initial_name="/World/ur5e", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        ur5e_robot_name = find_unique_string_name(
            initial_name="my_ur5e", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        
        return UR5eHandeye(prim_path = ur5e_prim_path,
                           name = ur5e_robot_name,
                           usd_path = ur5e_usd_path)

    
    def get_params(self) -> dict:
        params_representation = dict()
        if self.imported_objects is None:
            self.position, self.orientation = self._task_object.get_local_pose()
            self.task_object_name = self._task_object.name
        else:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.imported_objects_prim_path)
            matrix = omni.usd.get_world_transform_matrix(prim)
            translate = matrix.ExtractTranslation()
            rotation = matrix.ExtractRotationQuat()
            self.position = np.array([translate[0], translate[1], translate[2]],
                                     dtype=np.float32)
            self.orientation = np.array([rotation.imaginary[0],
                                         rotation.imaginary[1],
                                         rotation.imaginary[2],
                                         rotation.real],
                                         dtype=np.float32)
            self.task_object_name = prim.GetName()

        params_representation["task_object_position"] = {"value": self.position, "modifiable": True}
        params_representation["task_object_orientation"] = {"value": self.orientation, "modifiable": True}
        params_representation["task_object_name"] = {"value": self.task_object_name, "modifiable": False}
        params_representation["target_position"] = {"value": self._target_position, "modifiable": True}
        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation


    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        return {
            self.task_object_name: {
                "position": self.position,
                "orientation": self.orientation,
                "target_position": self._target_position,
            },
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            },
        }
    

