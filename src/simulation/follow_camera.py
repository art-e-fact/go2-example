import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni.kit.viewport.utility as vu
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation


class FollowCamera:
    """
    A simple follow camera that tracks a target articulation.
    Automatically turns off when the user moves the camera manually in the viewport.
    """

    def __init__(self, target_prim_path: str = "/World/Go2/Head_lower"):
        self.camera = Camera(
            prim_path="/World/camera",
            position=np.array([-0.1, 0.0, 0.0]),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 0, 0]), degrees=True
            ),
        )
        self.camera.set_focal_length(1.8)  # Same as the default perspective camera
        self.camera_location = np.array([0.0, 5.0, 2.0])
        self.target = SingleArticulation(target_prim_path)
        self.enabled = True
        self.last_camera_location = None
        self.last_camera_orientation = None

    def initialize(self):
        self.camera.initialize()
        viewport = vu.get_active_viewport()
        viewport.camera_path = "/World/camera"
        self.last_camera_location, self.last_camera_orientation = (
            self.camera.get_world_pose()
        )

    def set_camera_location(self, location: np.ndarray):
        self.camera_location = location

    def reset(self):
        self.last_camera_location = None
        self.last_camera_orientation = None
        self.enabled = True

    def update(self):
        if not self.enabled:
            return

        current_pos, current_rot = self.camera.get_world_pose()

        # Check if the camera has been moved manually
        if self.last_camera_location is not None and (
            not np.allclose(current_pos, self.last_camera_location, atol=1e-4)
            or not np.allclose(current_rot, self.last_camera_orientation, atol=1e-4)
        ):
            print("FollowCamera disabled due to manual camera movement.")
            self.enabled = False
            return

        target_loc, _ = self.target.get_world_pose()
        camera_rot = look_at_quat(self.camera_location, target_loc)
        self.camera.set_world_pose(self.camera_location, camera_rot)

        # Store the new pose
        self.last_camera_location = self.camera_location.copy()
        self.last_camera_orientation = camera_rot.copy()


def look_at_quat(
    source_pos: np.ndarray,
    target_pos: np.ndarray,
    up_vector: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> np.ndarray:
    """
    Calculates the 'look-at' quaternion for an Isaac Sim camera.

    The result is the orientation of a camera at `source_pos` that makes it
    "look at" `target_pos`, assuming the Isaac Sim camera convention where the
    local +X axis points forward.

    Args:
        source_pos: The position of the camera.
        target_pos: The position of the target to look at.
        up_vector: The world's "up" direction. Defaults to Z-up.

    Returns:
        A wxyz quaternion as a NumPy array.
    """
    # Isaac Sim camera's "forward" is its local +X axis.
    cam_forward = target_pos - source_pos

    if np.linalg.norm(cam_forward) < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])

    cam_forward = cam_forward / np.linalg.norm(cam_forward)

    # The camera's "up" is its local +Z axis.
    # We compute the "right" vector first, which is the camera's local -Y.
    cam_right = np.cross(cam_forward, up_vector)

    # Handle edge case where cam_forward is parallel to up_vector
    if np.linalg.norm(cam_right) < 1e-6:
        # If looking straight up or down, use a different temporary up vector
        temp_up = np.array([0.0, 1.0, 0.0])
        if np.abs(np.dot(cam_forward, temp_up)) > 0.999999:
            temp_up = np.array([1.0, 0.0, 0.0])
        cam_right = np.cross(cam_forward, temp_up)

    cam_right = cam_right / np.linalg.norm(cam_right)

    # Compute the true camera "up" (local +Z)
    cam_up = np.cross(cam_right, cam_forward)

    # Create a rotation matrix from the basis vectors.
    # Columns are the new axes in the world frame.
    # X_local -> cam_forward
    # Y_local -> -cam_right (since cam_right is -Y)
    # Z_local -> cam_up
    rotation_matrix = np.array([cam_forward, -cam_right, cam_up]).T

    try:
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    except ValueError as e:
        print(f"Error converting matrix to quaternion: {e}")
        print(f"Rotation matrix:\n{rotation_matrix}")
        print(f"Determinant: {np.linalg.det(rotation_matrix)}")
        return np.array([1.0, 0.0, 0.0, 0.0])

    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    return quat_wxyz
