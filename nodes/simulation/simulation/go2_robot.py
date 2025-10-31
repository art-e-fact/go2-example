from pathlib import Path
from typing import Optional

import msgs
import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController, config_loader
from scipy.spatial.transform import Rotation

from .height_scan import HeightScanGrid


class Go2Policy(PolicyController):
    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "go2",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        debug_vis: bool = False,
    ) -> None:
        """Initialize robot and load RL policy.

        For now, this class is used for both policy inference and control over DDS

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """
        if usd_path == None:
            usd_path = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)

        # TODO: load policy config from messages
        policy_path = (
            Path(__file__).resolve().parent.parent.parent
            / "policy_controller"
            / "policy_controller"
            / "policy"
        )
        env_path = policy_path / "env.yaml"
        if not env_path.exists():
            raise FileNotFoundError(f"Env config file not found at {env_path}")
        self.policy_env_params = config_loader.parse_env_config(str(env_path))

        self.height_scan_grid = HeightScanGrid(debug_vis=debug_vis)

    def is_on_back(self) -> bool:
        """Check if the robot is on its back.

        It does this by taking the robot's current orientation,
        and seeing where the robot's z-axis is pointing in world coordinates.
        """
        orientation_wxyz = self.robot.get_world_pose()[1]
        if orientation_wxyz is None:
            return False

        rotation = Rotation.from_quat(orientation_wxyz, scalar_first=True)
        robot_up = np.array([0.0, 0.0, 1.0])

        # Rotate the robot's up-vector by its current orientation
        world_up = rotation.apply(robot_up)

        # If the z-component of the world_up vector is negative, the robot is upside down.
        # We use a threshold to be robust to small tilts.
        return world_up[2] < -0.5

    def _compute_relative(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the relative linear and angular velocity, and gravity vector in the body frame."""
        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        # Compute yaw (rotation around Z-axis) from a (w, x, y, z) quaternion
        yaw = np.arctan2(
            2.0 * (q_IB[0] * q_IB[3] + q_IB[1] * q_IB[2]),
            1.0 - 2.0 * (q_IB[2] * q_IB[2] + q_IB[3] * q_IB[3]),
        )
        self.height_scan_grid.scan(pos_IB, yaw)

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        return lin_vel_b, ang_vel_b, gravity_b

    def compute_observations(self) -> msgs.Observations:
        lin_vel_b, ang_vel_b, gravity_b = self._compute_relative()
        return msgs.Observations(
            lin_vel=lin_vel_b,
            ang_vel=ang_vel_b,
            gravity=gravity_b,
            joint_positions=self.robot.get_joint_positions(),
            joint_velocities=self.robot.get_joint_velocities(),
            height_scan=self.height_scan_grid.get_height_data(),
        )

    def _report_all_hits(self, hit_info):
        return True

    def set_target_positions(self, target_positions: np.ndarray):
        """Set the target positions for the robot's joints.

        Args:
            target_positions (np.ndarray): The target joint positions.

        """
        target_positions = target_positions[:12]
        action = ArticulationAction(joint_positions=target_positions)
        self.robot.apply_action(action)
