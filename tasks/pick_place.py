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
from ur5 import UR5
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.manipulators.grippers import ParallelGripper
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import get_stage_units
import numpy as np
from typing import Optional


class PickPlace(tasks.PickPlace):
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
        name: str = "ur5_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        gripper_name: Optional[str] = None,
    ) -> None:
        self.gripper_name = gripper_name
        if cube_size is None:
            cube_size = np.array([0.0515, 0.0515, 0.0515]) / get_stage_units()
        if target_position is None:
            target_position = np.array([0.7, 0.7, cube_size[2] / 2.0])
            target_position[0] = target_position[0] / get_stage_units()
            target_position[1] = target_position[1] / get_stage_units()
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        return

    def set_robot(self) -> UR5:
        """[summary]

        Returns:
            UR5: [description]
        """
        # ur5_prim_path = find_unique_string_name(
        #     initial_name="/World/ur5", is_unique_fn=lambda x: not is_prim_path_valid(x)
        # )
        # ur5_robot_name = find_unique_string_name(
        #     initial_name="my_ur5", is_unique_fn=lambda x: not self.scene.object_exists(x)
        # )
        # self._ur5_robot = UR5(prim_path=ur5_prim_path, name=ur5_robot_name, attach_gripper=True, gripper_usd=self.gripper_name)
        # self._ur5_robot.set_joints_default_state(
        #     positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        # )
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        asset_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/ur5")

        gripper_usd = assets_root_path + "/Isaac/Robots/Robotiq/2F-140/2f140_instanceable.usd"
        add_reference_to_stage(usd_path=gripper_usd, prim_path="/World/ur5/tool0")
        
        gripper = ParallelGripper(
            # We chose the following values while inspecting the articulation
            end_effector_prim_path="/World/ur5/tool0/robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint", "right_outer_knuckle_joint"],
            joint_opened_positions=np.array([0.0, 0.0]),
            joint_closed_positions=np.array([0.628, -0.628]),
            action_deltas=np.array([0.05, 0.05]) / get_stage_units(),
        )
        # define the manipulator
        self._ur5_robot = SingleManipulator(
                prim_path="/World/ur5",
                name="ur5",
                end_effector_prim_name="tool0/robotiq_arg2f_base_link",
                gripper=gripper,
            )
        self._ur5_robot.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )
        return self._ur5_robot

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """[summary]

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """
        tasks.PickPlace.pre_step(self, time_step_index=time_step_index, simulation_time=simulation_time)
        self._ur5_robot.gripper.update()
        return
