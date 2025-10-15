from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from scipy.spatial.transform import Rotation

from .height_scan import HeightScanGrid

if TYPE_CHECKING:
    from dds.go2_robot_dds import Go2RobotDDS


class ControlDtStepper:
    """
    Accumulates dt and tell when should the policy compute new actions
    """

    def __init__(self, policy_dt, policy_decimation):
        self.control_timestep = policy_dt * policy_decimation
        self.elapsed = None

    def should_step(self, dt) -> bool:
        # Force the first step immediately
        if self.elapsed is None:
            self.elapsed = 0
            return True

        self.elapsed += dt
        if self.elapsed < self.control_timestep - 1e-6:
            return False

        if self.elapsed >= self.control_timestep:
            print(f"Warning: Control timestep exceeded: {self.elapsed:.3f} seconds")
        self.elapsed = 0

        return True

class ControlCountingStepper:
    """
    Counts the number of calls and tells when to step based on decimation
    """

    def __init__(self, policy_dt, policy_decimation):
        self.decimation = policy_decimation
        self.counter = 0

    def should_step(self, dt) -> bool:
        if self.counter % self.decimation == 0:
            self.counter += 1
            return True
        self.counter += 1
        return False


class Go2Policy(PolicyController):
    """The Go2 quadruped"""

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
        """
        Initialize robot and load RL policy.

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

        demo_policy_path = Path(__file__).resolve().parent.parent.parent / "demo_policy"
        self.load_policy(
            str(demo_policy_path / "policy.pt"),
            str(demo_policy_path / "env.yaml"),
        )
        self._action_scale = 0.2
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        # self._control_stepper = ControlDtStepper(
        #     policy_dt=self._dt,
        #     policy_decimation=self._decimation,
        # )
        self._control_stepper = ControlCountingStepper(
            policy_dt=self._dt,
            policy_decimation=self._decimation,
        )

        self.height_scan_grid = HeightScanGrid(debug_vis=debug_vis)

    def reset(self):
        super().reset()
        self._previous_action = np.zeros(12)
        self._policy_counter = 0
        self._control_stepper.elapsed = None

    def is_on_back(self) -> bool:
        """
        Checks if the robot is on its back.
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

    def _compute_relative(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the relative linear and angular velocity, and gravity vector in the body frame.
        """
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

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_b, ang_vel_b, gravity_b = self._compute_relative()

        obs = np.zeros(202)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self.default_pos
        obs[24:36] = current_joint_vel
        # Previous Action
        obs[36:48] = self._previous_action
        # Height Scan (154)
        height_data = self.height_scan_grid.get_height_data()
        assert height_data.shape == (154,), (
            f"Height data shape mismatch: {height_data.shape}"
        )
        obs[48:202] = height_data

        return obs

    def write_robot_dds_state(self, go2_dds: "Go2RobotDDS"):
        lin_vel_b, ang_vel_b, gravity_b = self._compute_relative()
        robot_velocity = self.robot.get_linear_velocity()
        robot_position, quaternion = self.robot.get_world_pose()
        go2_dds.write_robot_state(
            quaternion.tolist(),
            lin_vel_b.tolist(),
            ang_vel_b.tolist(),
            gravity_b.tolist(),
            self.robot.get_joint_positions().tolist(),
            self.robot.get_joint_velocities().tolist(),
            self.robot.get_applied_joint_efforts().tolist(),
            robot_position.tolist(),
            robot_velocity.tolist(),
        )

    def read_targets_from_dds(self, go2_dds: "Go2RobotDDS"):
        """
        Read the target positions from the DDS.

        Argument:
        go2_dds (Go2RobotDDS) -- The DDS instance to read from.

        """
        target_positions = go2_dds.get_target_positions()
        if target_positions is not None:
            self._set_target_positions(target_positions)

    def report_all_hits(self, hit_info):
        return True

    def forward(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        # if self._policy_counter % (self._decimation * 2) == 0: # TODO decide which on makes more sense
        if self._control_stepper.should_step(dt):
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)

            # raise if any of the action is nan or inf
            if np.isnan(self.action).any() or np.isinf(self.action).any():
                raise ValueError(f"Invalid action: {self.action}")
            self._previous_action = self.action.copy()

        action = ArticulationAction(
            joint_positions=self.default_pos + (self.action * self._action_scale)
        )

        self.robot.apply_action(action)

        self._policy_counter += 1

    def _set_target_positions(self, target_positions: np.ndarray):
        """
        Set the target positions for the robot's joints.

        Args:
            target_positions (np.ndarray): The target joint positions.
        """
        target_positions = target_positions[:12]
        action = ArticulationAction(joint_positions=target_positions)
        self.robot.apply_action(action)
