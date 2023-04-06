# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.core.articulations import Articulation
from rmpflow_controller import RMPFlowController
from typing import Optional, List

from omni.isaac.core.controllers import BaseController
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import numpy as np
import typing
from omni.isaac.manipulators.grippers.gripper import Gripper

class ReachTargetController(manipulators_controllers.PickPlaceController):
    """ 
        A simple pick and place state machine for tutorials

        Each phase runs for 1 second, which is the internal time of the state machine

        Dt of each phase/ event step is defined

        - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.

        Args:
            name (str): Name id of the controller
            cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
            gripper (Gripper): a gripper controller for open/ close actions.
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
    """
    """[summary]

        Args:
            name (str): [description]
            gripper (ParallelGripper): [description]
            robot_articulation (Articulation): [description]
            end_effector_initial_height (Optional[float], optional): [description]. Defaults to None.
            events_dt (Optional[List[float]], optional): [description]. Defaults to None.
        """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: Articulation,
        end_effector_initial_height: Optional[float] = None,
        events_dt: Optional[List[float]] = None,
    ) -> None:
        if events_dt is None:
            # events_dt = [0.008, 0.005, 1, 0.05, 0.05, 0.05, 0.0025, 1, 0.008, 0.08]
            events_dt = [0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            end_effector_initial_height=end_effector_initial_height,
            events_dt=events_dt,
        )
        return


    def forward(
        self,
        picking_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        theta: np.int8 = None,
    ) -> ArticulationAction:
        """Runs the controller one step.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray):  The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot. 12
            end_effector_offset (typing.Optional[np.ndarray], optional): offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): end effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])
        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)
    
        ## reach target
        self._current_target_x = picking_position[0]
        self._current_target_y = picking_position[1]
        self._h0 = picking_position[2]

        interpolated_xy = self._get_interpolated_xy(
            0, 0, self._current_target_x, self._current_target_y
        )
        position_target = np.array(
            [
                interpolated_xy[0] + end_effector_offset[0],
                interpolated_xy[1] + end_effector_offset[1],
                self._h0 + end_effector_offset[2],
            ]
        )
        if end_effector_orientation is None:
            # end_effector_orientation = euler_angles_to_quat(np.array([0, theta * np.pi / 360, 0]))
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, theta * 2 * np.pi / 360]))

        target_joint_positions = self._cspace_controller.forward(
            target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
        )

        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0
        
        return target_joint_positions
        

