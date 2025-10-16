from typing import Optional
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import SingleXFormPrim
from simulation.waypoint_mission import WaypointMission, WaypointStatus
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils

from simulation.video_writer import VideoWriter

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

        # Topdown camera
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

        self.head_video_writer = VideoWriter(
            camera=self.head_camera,
            output_path="outputs/artefacts/head_camera.mp4",
            width=self.head_camera_width,
            height=self.head_camera_height,
            framerate=self.framerate,
        )
        self.topdown_video_writer = VideoWriter(
            camera=self.topdown_camera,
            output_path="outputs/artefacts/topdown_camera.mp4",
            width=self.topdown_cam_res,
            height=self.topdown_cam_res,
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
        self.topdown_video_writer.initialize()


    def link_waypoint_mission(self, waypoint_mission: WaypointMission):
        self.waypoint_mission = waypoint_mission
        self.calibrate_topdown_projection()

    def capture_frames(self):
        self.head_video_writer.capture_frame()
        self.topdown_video_writer.capture_frame()
        self.follow_video_writer.capture_frame()

    def close(self):
        self.head_video_writer.close()
        self.topdown_video_writer.close()
        self.follow_video_writer.close()
