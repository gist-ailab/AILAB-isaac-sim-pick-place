# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import omni.isaac.core.tasks as tasks
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name

# add necessary directories to sys.path
import sys, os
from pathlib import Path
current_dir = os.path.dirname(os.path.realpath(__file__))
directory = Path(current_dir).parent
sys.path.append(str(directory))

from robots.ur5e_handeye import UR5eHandeye
import random
import numpy as np
from typing import Optional


class SetUpUR5e(tasks.BaseTask):
    """[summary]

        Args:
            name (str, optional): [description]. Defaults to "ur5_pick_place".
        """

    def __init__(
        self,
        name: str = "set_up_ur5e",
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, )


        return


    def set_up_scene(self, scene: Scene) -> None:
        """[summary]
        The robot added to the scene.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        scene.add_default_ground_plane()
                
        self._robot = self.set_robot()
        scene.add(self._robot)

        # self._move_task_objects_to_their_frame()
        return


    def set_robot(self) -> UR5eHandeye:
        """[summary]

        Returns:
            UR5e: [description]
        """
        working_dir = os.path.dirname(os.path.realpath(__file__))   # same directory with this code
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

        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}
        return params_representation


    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()

        observation_dict = dict()
        
        observation_dict = {
                            self._robot.name: {"joint_positions": joints_state.positions,
                                                "end_effector_position": end_effector_position,
                                                },
                            }

        return observation_dict
    


class SetUpUR5eObject(tasks.BaseTask):
    """[summary]

        Args:
            name (str, optional): [description]. Defaults to "ur5_pick_place".
        """

    def __init__(
        self,
        name: str = "set_up_ur5e",
        object_position: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, )

        cube_prim_path = "/World/Cube"
        cube_name = "cube1"
        if object_position is None:
            pos_x = random.uniform(0.3, 0.6)
            pos_y = random.uniform(0.3, 0.6)
            pos_z = 0.1
            self.object_position = np.array([[pos_x, pos_y, pos_z]])

        self._object = DynamicCuboid(
            prim_path = cube_prim_path,
            name = cube_name,
            position = self.object_position,
            color = np.array([0, 0, 1]),
            size = 0.04,
            mass = 0.01,
        )

        return


    def set_up_scene(self, scene: Scene) -> None:
        """[summary]
        The robot added to the scene.

        Args:
            scene (Scene): [description]
        """
        self._scene = scene
        scene.add_default_ground_plane()
                
        self._robot = self.set_robot()
        scene.add(self._robot)

        self._task_objects = {self._object.name: scene.add(self._object)}
        self._move_task_objects_to_their_frame()
        return


    def set_robot(self) -> UR5eHandeye:
        """[summary]

        Returns:
            UR5e: [description]
        """
        working_dir = os.path.dirname(os.path.realpath(__file__))   # same directory with this code
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

        self.object_position, self.object_orientation = self._object.get_local_pose()
        params_representation["object_position"] = {"value": self.object_position, "modifiable": False}
        params_representation["object_orientation"] = {"value": self.object_orientation, "modifiable": False}
        params_representation["object_name"] = {"value": self._object.name, "modifiable": False}

        params_representation["robot_name"] = {"value": self._robot.name, "modifiable": False}

        return params_representation


    def get_observations(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()

        observation_dict = dict()
        
        observation_dict = {
                            self._object.name: {"object_position": self.object_position,
                                                    "object_orientation": self.object_orientation,
                                                    },
                            self._robot.name: {"joint_positions": joints_state.positions,
                                                "end_effector_position": end_effector_position,
                                                },
                            }

        return observation_dict
    