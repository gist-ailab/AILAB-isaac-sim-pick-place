# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.universal_robots.controllers import RMPFlowController
import numpy as np
from typing import Optional, List


class PickPlaceController(manipulators_controllers.PickPlaceController):
    """[summary]

        Args:
            name (str): [description]
            surface_gripper (SurfaceGripper): [description]
            robot_articulation(Articulation): [description]
            events_dt (Optional[List[float]], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            # events_dt = [0.01, 0.0035, 0.01, 1.0, 0.008, 0.005, 0.005, 1, 0.01, 0.08]
            # events_dt = [0.008, 0.005, 1, 0.05, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
            events_dt = [0.008, 0.01, 1, 0.08, 0.1, 0.05, 0.005, 1, 0.01, 0.1]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation, attach_gripper=True
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        return

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
        end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            picking_position (np.ndarray): [description]
            placing_position (np.ndarray): [description]
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_orientation (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, np.pi]))

        return super().forward(
            picking_position,
            placing_position,
            current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )


    # def forward(
    #     self,
    #     picking_position: np.ndarray,
    #     placing_position: np.ndarray,
    #     current_joint_positions: np.ndarray,
    #     end_effector_offset: typing.Optional[np.ndarray] = None,
    #     end_effector_orientation: typing.Optional[np.ndarray] = None,
    # ) -> ArticulationAction:
    #     """Runs the controller one step.

    #     Args:
    #         picking_position (np.ndarray): The object's position to be picked in local frame.
    #         placing_position (np.ndarray):  The object's position to be placed in local frame.
    #         current_joint_positions (np.ndarray): Current joint positions of the robot.
    #         end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
    #         end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

    #     Returns:
    #         ArticulationAction: action to be executed by the ArticulationController
    #     """
    #     if end_effector_offset is None:
    #         end_effector_offset = np.array([0, 0, 0])
    #     if self._pause or self.is_done():
    #         self.pause()
    #         target_joint_positions = [None] * current_joint_positions.shape[0]
    #         return ArticulationAction(joint_positions=target_joint_positions)
    #     if self._event == 2:
    #         target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
    #     elif self._event == 3:
    #         target_joint_positions = self._gripper.forward(action="close")
    #     elif self._event == 7:
    #         target_joint_positions = self._gripper.forward(action="open")
    #     else:
    #         if self._event in [0, 1]:
    #             self._current_target_x = picking_position[0]
    #             self._current_target_y = picking_position[1]
    #             self._h0 = picking_position[2]
    #         interpolated_xy = self._get_interpolated_xy(
    #             placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
    #         )
    #         target_height = self._get_target_hs(placing_position[2])
    #         position_target = np.array(
    #             [
    #                 interpolated_xy[0] + end_effector_offset[0],
    #                 interpolated_xy[1] + end_effector_offset[1],
    #                 target_height + end_effector_offset[2],
    #             ]
    #         )
    #         if end_effector_orientation is None:
    #             end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi / 2.0, np.pi]))
    #         target_joint_positions = self._cspace_controller.forward(
    #             target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
    #         )
    #     self._t += self._events_dt[self._event]
    #     if self._t >= 1.0:
    #         self._event += 1
    #         self._t = 0
    #     return target_joint_positions