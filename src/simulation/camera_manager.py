from pathlib import Path
from typing import Optional
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import SingleXFormPrim
from simulation.waypoint_mission import WaypointMission, WaypointStatus
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from PIL import Image

from simulation.video_writer import VideoWriter
import omni 

WAYPOINT_COLOR_MAP = {
    WaypointStatus.INACTIVE: (255, 255, 0),
    WaypointStatus.ACTIVE: (0, 0, 255),
    WaypointStatus.COMPLETED: (0, 255, 0),
}


def get_camera_rgb(camera: Camera):
    rgba = camera.get_rgba()
    if rgba is None:
        print("Warning: No image received from camera")
        return None
    if len(rgba.shape) != 3:
        print(f"Warning: Image has unexpected shape {rgba.shape}, expected (H,W,3)")
        return None

    rgb = rgba[:, :, :3]
    return rgb


class CameraManager:
    def __init__(self):
        self.head_camera_width = 1080 // 4
        self.head_camera_height = 720 // 4
        self.framerate = 20
        self.head_camera = Camera(
            prim_path="/World/Go2/Head_upper/camera",
            translation=np.array([0.04, 0.0, 0.021]),
            frequency=self.framerate,
            resolution=(self.head_camera_width, self.head_camera_height),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 0, 0]), degrees=True
            ),
        )

        self.head_camera.set_focal_length(3.0)
        self.head_camera.set_clipping_range(0.01, 1000000000.0)

        # Topdown camera (only used to taka a picture of the whole scene)
        self.topdown_center = np.array([0.0, 0.0])
        self.topdown_box_size = 100.0
        self.topdown_cam_res = 1200
        self.topdown_cam_height = 3.3
        self.topdown_camera = Camera(
            prim_path="/World/topdown",
            position=np.array([0.0, 0.0, 5.0]),
            frequency=self.framerate,
            resolution=(self.topdown_cam_res, self.topdown_cam_res),
        )
        self.topdown_camera.set_world_pose(
            np.array([0.0, 0.0, 5.0]), [1, 0, 0, 0], camera_axes="usd"
        )
        self.topdown_camera.set_projection_mode("orthographic")
        self.topdown_wait_frames = 15
        self.topdown_image_countdown = 0

        self.head_video_writer = VideoWriter(
            camera=self.head_camera,
            output_path="outputs/artefacts/head_camera.mp4",
            width=self.head_camera_width,
            height=self.head_camera_height,
            framerate=self.framerate,
        )
        self.follow_cameara = Camera("/World/camera")
        self.follow_video_writer = VideoWriter(
            camera=self.follow_cameara,
            output_path="outputs/artefacts/follow_camera.mp4",
            width=1080 // 4,
            height=720 // 4,
            framerate=self.framerate,
        )
        self.topdown_snapshot_path = Path("outputs/artefacts/topdown_camera.png")

    def calibrate_topdown_projection(self):
        waypoints = [wp.get_position()[0][:2] for wp in self.waypoint_mission.waypoints]
        if len(waypoints) > 1:
            self.topdown_center = np.mean(waypoints, axis=0)
            self.topdown_box_size = (
                np.max(np.abs(waypoints - self.topdown_center), axis=0).max() * 2 * 1.2
            )
        else:
            self.topdown_center = np.array([0.0, 0.0])
            self.topdown_box_size = 10.0

        self.topdown_camera.set_world_pose(
            np.array(
                [
                    self.topdown_center[0],
                    self.topdown_center[1],
                    self.topdown_cam_height,
                ]
            ),
            [1, 0, 0, 0],
            camera_axes="usd",
        )
        self.topdown_camera.set_horizontal_aperture(self.topdown_box_size)

    def initialize(self):
        self.head_camera.initialize()
        self.topdown_camera.initialize()
        self.head_video_writer.initialize()

        # Reset the topdown image countdown
        self.topdown_image_countdown = self.topdown_wait_frames

    def link_waypoint_mission(self, waypoint_mission: WaypointMission):
        self.waypoint_mission = waypoint_mission
        self.calibrate_topdown_projection()

    def capture_frames(self):
        self.head_video_writer.capture_frame()
        self.follow_video_writer.capture_frame()
        self.capture_topdown_snapshot()

    def capture_topdown_snapshot(self):
        """Waits untils the topdown camera is ready and captures a single snapshot."""

        # If countdown is zero, we have already taken the snapshot
        if self.topdown_image_countdown <= 0:
            return

        # Enable the topdown camera to take a shot for the map
        self.topdown_camera._render_product.hydra_texture.set_updates_enabled(True)

        # Use camera lighting
        action_registry = omni.kit.actions.core.get_action_registry()
        action = action_registry.get_action(
            "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
        )
        action.execute()

        rgb = get_camera_rgb(self.topdown_camera)
        if rgb is not None:
            self.topdown_image_countdown -= 1

            if self.topdown_image_countdown == 0:
                # Save the image
                img = Image.fromarray(rgb)
                img.save(self.topdown_snapshot_path)

                # Switch to stage lighting
                action = action_registry.get_action(
                    "omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage"
                )
                action.execute()
                # Disable further updates to the topdown camera
                self.topdown_camera._render_product.hydra_texture.set_updates_enabled(False)

    def close(self):
        self.head_video_writer.close()
        self.topdown_video_writer.close()
        self.follow_video_writer.close()
