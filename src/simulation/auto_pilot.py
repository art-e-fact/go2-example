import numpy as np
from isaacsim.core.prims import SingleXFormPrim
from scipy.spatial.transform import Rotation

from .waypoint_mission import WaypointMission


class AutoPilot:
    def __init__(self, waypoint_mission: WaypointMission):
        self.waypoint_mission = waypoint_mission

    def advance(self):
        """
        Calculate control commands to navigate to the next waypoint.

        Returns:
            np.ndarray: Command vector [linear_x, linear_y, angular_z]
        """
        # Skip if mission is complete
        if self.waypoint_mission.is_complete() or not self.waypoint_mission.waypoints:
            return np.zeros(3)

        # Get current waypoint and its position
        current_index = self.waypoint_mission.current_index
        if current_index >= len(self.waypoint_mission.waypoints):
            return np.zeros(3)

        waypoint = self.waypoint_mission.waypoints[current_index]
        waypoint_pos, _ = waypoint.get_position()

        # Get robot position
        robot_xform = SingleXFormPrim(str(self.waypoint_mission.robot_path))
        robot_pos, robot_quat = robot_xform.get_world_pose()

        # Calculate 2D direction vector (ignoring height)
        direction = waypoint_pos[:2] - robot_pos[:2]
        distance = np.linalg.norm(direction)

        # If we're very close to the waypoint, just stop
        if distance < 0.1:
            return np.zeros(3)

        # Normalize direction vector
        if distance > 0:
            direction = direction / distance

        # Use scipy Rotation to get forward vector from quaternion
        # Note: robot_quat is [w,x,y,z] but scipy expects [x,y,z,w]
        rot = Rotation.from_quat(
            [robot_quat[1], robot_quat[2], robot_quat[3], robot_quat[0]]
        )

        # Apply rotation to the forward unit vector [1,0,0]
        forward_3d = rot.apply([1, 0, 0])
        forward = forward_3d[:2]  # Take just x,y components for 2D navigation
        forward = (
            forward / np.linalg.norm(forward)
            if np.linalg.norm(forward) > 0
            else np.array([1, 0])
        )

        # Calculate angle between robot forward and target direction
        dot = np.clip(np.dot(forward, direction), -1.0, 1.0)
        angle = np.arccos(dot)
        cross = np.cross(np.append(forward, 0), np.append(direction, 0))[2]
        if cross < 0:
            angle = -angle

        # Calculate command velocities
        linear_speed = 1.0  # Maximum linear speed
        angular_speed = 2.0  # Maximum angular speed

        # Scale down linear speed when turning sharply
        turn_factor = abs(angle) / np.pi
        linear_speed *= 1.0 - 0.7 * turn_factor

        # Create command vector [linear_x, linear_y, angular_z]
        command = np.zeros(3)
        command[0] = linear_speed * (
            1 - 0.8 * min(1.0, turn_factor)
        )  # Forward velocity
        command[2] = angular_speed * np.clip(angle, -1.0, 1.0)  # Angular velocity

        return command
