import numpy as np
from scipy.spatial.transform import Rotation

def compute_command(robot_position: np.ndarray, robot_quaternion: np.ndarray, goal_position: np.ndarray) -> np.ndarray:
    """Calculate control command based on the robot pose and the goal position.

    It will drive straight to the goal without considering obstacles.

    Args:
        robot_position: Current position of the robot [x, y, z]
        robot_quaternion: Current orientation as quaternion [w, x, y, z]
        goal_position: Target position [x, y, z]

    Returns:
        np.ndarray: Command vector [linear_x, linear_y, angular_z]
    """
    # Calculate 2D direction vector (ignoring height)
    direction = goal_position[:2] - robot_position[:2]
    distance = np.linalg.norm(direction)

    # If we're very close to the goal, just stop
    if distance < 0.1:
        return np.zeros(3)

    # Normalize direction vector
    if distance > 0:
        direction = direction / distance

    # Use scipy Rotation to get forward vector from quaternion
    # Note: robot_quaternion is [w,x,y,z] but scipy expects [x,y,z,w]
    rot = Rotation.from_quat(
        [robot_quaternion[1], robot_quaternion[2], robot_quaternion[3], robot_quaternion[0]]
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