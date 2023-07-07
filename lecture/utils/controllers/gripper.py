from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.articulations import Articulation
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
# import omni.isaac.manipulators.controllers as manipulators_controllers
from omni.isaac.universal_robots.controllers import RMPFlowController
import numpy as np
from typing import Optional, List

from . import gripper_controller

class GripperController(gripper_controller.GripperController):
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
            events_dt = [0.01]
        gripper_controller.GripperController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation, attach_gripper=True
            ),
            gripper=gripper,
            events_dt=events_dt,
        )
        return

    def open(
        self,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.23])


        return super().open(
            current_joint_positions,
            end_effector_offset=end_effector_offset,
        )
    
    def close(
        self,
        current_joint_positions: np.ndarray,
        end_effector_offset: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """[summary]

        Args:
            current_joint_positions (np.ndarray): [description]
            end_effector_offset (Optional[np.ndarray], optional): [description]. Defaults to None.

        Returns:
            ArticulationAction: [description]
        """
        
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0.23])


        return super().close(
            current_joint_positions,
            end_effector_offset=end_effector_offset,
        )

