from typing import Optional
import av
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import SingleXFormPrim
from scipy.spatial.transform import Rotation
from simulation.waypoint_mission import WaypointMission, WaypointStatus
from simulation.rtf_calculator import RtfCalculator
from datetime import datetime
import time
import uuid

WAYPOINT_COLOR_MAP = {
    WaypointStatus.INACTIVE: (255, 255, 0),
    WaypointStatus.ACTIVE: (0, 0, 255),
    WaypointStatus.COMPLETED: (0, 255, 0),
}

import omni.kit.actions.core


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


CONTROLLER_GUIDE = """
```sh
Controller Quick Guide
                 
    Left Stick (Move)           Right Stick (Yaw)           
           +X                     
            ▲                           ▲
(-Y)    ◀   ●   ▶   (+Y)    (-Z)    ◀   ●   ▶   (+Z)           
            ▼                           ▼
           -X                    
```
"""


class RerunLogger:
    def __init__(self, robot_path: str, use_video_stream: bool):
        self.robot_path = robot_path
        self.robot_xform = SingleXFormPrim(str(self.robot_path) + "/base")
        self.use_video_stream = use_video_stream
        self.width = 1080 // 4
        self.height = 720 // 4
        self.framerate = 20
        self.camera = Camera(
            prim_path="/World/Go2/Head_upper/camera",
            translation=np.array([0.04, 0.0, 0.021]),
            frequency=self.framerate,
            resolution=(self.width, self.height),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 0, 0]), degrees=True
            ),
        )
        self.scene_name = "Unknown Scene"

        # TODO figure out how to compute correctly the 120 degrees FOV in Isaac
        self.camera.set_focal_length(3.0)
        self.camera.set_clipping_range(0.01, 1000000000.0)

        self.robot_poses = []
        self.rec_id = None
        self.mission_start_time = None

        # Topdown camera
        self.topdown_center = np.array([0.0, 0.0])
        self.topdown_box_size = 100.0
        self.topdown_cam_res = 1200
        self.topdown_cam_height = 3.3  # This works with the hospital scene. We should make this a parameter once we have multiple scenes with a roof
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

    def xy_to_topdown(self, xy_coords: np.ndarray):
        """Convert from world XY coordinates to pixel coordinates in the topdown camera image."""
        # Return empty if no coordinates
        if xy_coords.shape[0] == 0:
            return xy_coords
        scale = self.topdown_cam_res / self.topdown_box_size
        tranformed = (xy_coords - self.topdown_center) * scale
        # Flip Y axis
        tranformed[:, 1] *= -1
        return tranformed + self.topdown_cam_res / 2

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

    def initialize(self, rrd_path: Optional[str] = None):
        self.rec_id = str(uuid.uuid4())[:8]
        self.mission_start_time = time.time()
        rr.init("Go2", spawn=False, recording_id=self.rec_id)
        if rrd_path:
            print(f"Saving Rerun recording to {rrd_path}\n\n\n\n\n\n\n\n\n\n\n\n\n")
            rr.save(rrd_path)
        else:
            print("Streaming Rerun recording over gRPC\n\n\n\n\n\n\n\n\n\n\n\n\n")
            rr.connect_grpc()

        self.robot_poses = []

        self.camera.initialize()
        self.topdown_camera.initialize()
        self.topdown_image_countdown = self.topdown_wait_frames
        # Enable the topdown camera to take a shot for the map
        self.topdown_camera._render_product.hydra_texture.set_updates_enabled(True)

        self.scene_name = "Unknown Scene"

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.TextDocumentView(
                    origin="report",
                    name="Mission Report",
                ),
                rrb.Spatial2DView(
                    origin="/go2/topdown",
                    name="Topdown View",
                ),
                rrb.Vertical(
                    rrb.TextDocumentView(
                        origin="guide",
                        name="Controller Guide / コントローラー早見表",
                    ),
                    rrb.Spatial2DView(
                        origin="/go2/head_camera",
                        name="Head Camera View",
                    ),
                ),
            ),
            collapse_panels=True,
        )

        rr.send_blueprint(blueprint)

        rr.log(
            "guide",
            rr.TextDocument(CONTROLLER_GUIDE, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )

        if self.use_video_stream:
            rr.log(
                "go2/head_camera", rr.VideoStream(codec=rr.VideoCodec.H264), static=True
            )
            av.logging.set_level(av.logging.VERBOSE)
            self.container = av.open(
                "/dev/null", "w", format="h264"
            )  # Use AnnexB H.264 stream.
            self.stream = self.container.add_stream("libx264", rate=20)
            self.stream.width = self.width
            self.stream.height = self.height
            # (#10090): Rerun Video Streams don't support b-frames yet.
            # Note that b-frames are generally not recommended for low-latency streaming and may make logging more complex.
            self.stream.max_b_frames = 0

    def link_waypoint_mission(self, waypoint_mission: WaypointMission):
        self.waypoint_mission = waypoint_mission
        self.calibrate_topdown_projection()

    def link_rtf_calculator(self, rtf_calculator: RtfCalculator):
        self.rtf_calculator = rtf_calculator

    def set_scene_name(self, scene_name: str):
        self.scene_name = scene_name

    def log(self):
        # Skip recording the first few frames while the robot is stabilizing
        if self.topdown_image_countdown <= 0:
            self.robot_poses.append(self.robot_xform.get_world_pose())
        self.log_camera()
        self.log_topdown()
        self.log_report()

    def log_report(self):
        # --- DATA PREPARATION ---
        mission = self.waypoint_mission
        if not mission:
            return

        # Overall Summary
        waypoints_reached = mission.current_index
        total_waypoints = len(mission.waypoints)

        total_distance = 0
        if len(self.robot_poses) > 1:
            positions = np.array([p for p, o in self.robot_poses])
            total_distance = np.sum(
                np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
            )

        elapsed_time = time.time() - self.mission_start_time

        avg_speed = total_distance / elapsed_time if elapsed_time > 0 else 0

        # Mission Status
        if waypoints_reached == total_waypoints and total_waypoints > 0:
            status_text = "Mission Complete"
        else:
            status_text = "In Progress"

        # Progress Bar
        progress_percent = (
            (waypoints_reached / total_waypoints) * 100 if total_waypoints > 0 else 0
        )
        progress_bar = f"Mission Progress: {int(progress_percent)}%"

        # Waypoint Details Table
        wp_table_rows = []
        for i, wp in enumerate(mission.waypoints):
            status = mission.get_waypoint_status(wp)

            if status == WaypointStatus.COMPLETED:
                status_label = "COMPLETED"
            elif status == WaypointStatus.ACTIVE:
                status_label = "ACTIVE"
            else:
                status_label = "INACTIVE"

            pos, _ = wp.get_position()
            pos_str = f"`({pos[0]:.1f}, {pos[1]:.1f})`"

            dist_to_next_str = "-"
            if i < total_waypoints - 1:
                next_pos, _ = mission.waypoints[i + 1].get_position()
                dist = np.linalg.norm(pos[:2] - next_pos[:2])
                dist_to_next_str = f"{dist:.1f}m"

            time_to_reach_str = "-"
            if hasattr(mission, "waypoint_completed_times") and i < len(
                mission.waypoint_completed_times
            ):
                # Calculate duration between waypoints
                start_time = (
                    mission.mission_start_time
                    if i == 0
                    else mission.waypoint_completed_times[i - 1]
                )
                end_time = mission.waypoint_completed_times[i]
                duration = end_time - start_time
                time_to_reach_str = f"{duration:.0f}s"

            wp_table_rows.append(
                f"| {status_label.ljust(9)} | {str(i + 1).ljust(7)} | {pos_str.ljust(22)} | {dist_to_next_str.ljust(16)} | {time_to_reach_str.ljust(13)} |"
            )

        wp_table_str = "\n".join(wp_table_rows)

        if self.waypoint_mission.is_complete():
            test_succeeded = (
                "\n\n# The robot successfully completed the waypoint mission!"
            )
        else:
            test_succeeded = ""

        if self.scene_name.lower() == "hospital_staircase":
            jouer_logo = "_**Provided by:**_\n\n![Jouer](https://drive.google.com/uc?export=view&id=1UwTWjl8hO45BmhFim6dKwWSS0BLwcY_n)"
        else:
            jouer_logo = ""

        # --- MARKDOWN REPORT ---
        report_md = f"""
![Artefacts](https://artefacts.com/images/artefacts_logo_web.svg)


# Test Suite
## Waypoint Mission: {self.scene_name} 
{jouer_logo}


**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Test Run ID:** `{self.rec_id}`
**Status:** {status_text}

---

## 📊 Overall Summary

| Metric                  | Value      |
| ----------------------- | ---------- |
| Waypoints Reached       | {waypoints_reached} / {total_waypoints}      |
| Total Distance          | {total_distance:.1f}m      |
| Elapsed Time            | `{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}`    |
| Average Speed           | {avg_speed:.2f} m/s   |
| Real-Time Factor (RTF)  | {self.rtf_calculator.rtf if hasattr(self, "rtf_calculator") else "N/A":.2f}      |

{progress_bar}

---

## Waypoint Details

| Status     | Waypoint | Target Position (x, y) | Distance to Next | Time to Reach |
| ---------- | -------- | ---------------------- | ---------------- | ------------- |
{wp_table_str}
{test_succeeded}
"""

        rr.log(
            "report",
            rr.TextDocument(
                report_md.strip(),
                media_type=rr.MediaType.MARKDOWN,
            ),
        )

    def log_topdown(self):
        if not hasattr(self, "waypoint_mission"):
            return
        if self.topdown_image_countdown > 0:
            rgb = get_camera_rgb(self.topdown_camera)
            if rgb is not None:
                self.topdown_image_countdown -= 1
                action_registry = omni.kit.actions.core.get_action_registry()
                # switches to camera lighting
                action = action_registry.get_action(
                    "omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera"
                )
                action.execute()

                if self.topdown_image_countdown == 0:
                    rr.log("go2/topdown/image", rr.Image(rgb), static=True)

                    # switches to stage lighting
                    action = action_registry.get_action(
                        "omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage"
                    )
                    action.execute()
                    # Disable further updates to the topdown camera
                    self.topdown_camera._render_product.hydra_texture.set_updates_enabled(
                        False
                    )

        rr.log(
            "go2/topdown/position",
            rr.LineStrips2D(
                self.xy_to_topdown(
                    np.array([[t[0], t[1]] for t, _ in self.robot_poses])
                )
            ),
        )

        if len(self.robot_poses) > 0:
            latest_pos, latest_orn = self.robot_poses[-1]
            origin = self.xy_to_topdown(np.array([latest_pos[:2]]))

            # Convert quaternion to euler angles to get yaw
            yaw = Rotation.from_quat(latest_orn, scalar_first=True).as_euler("xyz")[2]

            # Create a vector from yaw, flip y because of topdown projection
            vec = np.array([np.cos(yaw), -np.sin(yaw)])

            rr.log(
                "go2/topdown/robot_arrow",
                rr.Arrows2D(
                    origins=origin,
                    vectors=vec * 50,  # Scale vector for visibility
                    colors=[255, 100, 100],
                    radii=rr.Radius.ui_points(10.0),
                ),
            )

        wp_positions = self.xy_to_topdown(
            np.array(
                [wp.get_position()[0][:2] for wp in self.waypoint_mission.waypoints]
            )
        )
        wp_colors = [
            WAYPOINT_COLOR_MAP[self.waypoint_mission.get_waypoint_status(wp)]
            for wp in self.waypoint_mission.waypoints
        ]
        if hasattr(self, "waypoint_mission"):
            rr.log(
                "go2/topdown/waypoints",
                rr.Points2D(
                    wp_positions,
                    colors=wp_colors,
                    radii=rr.Radius.ui_points(25.0),
                ),
            )

    def log_camera(self):
        # TODO Should we skip frames when log is called to frequently?
        rgb = get_camera_rgb(self.camera)
        if rgb is None:
            return
        if self.use_video_stream:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in self.stream.encode(frame):
                if packet.pts is None:
                    continue
                # rr.set_time("time", duration=float(packet.pts * packet.time_base)) # TODO should we set the time manually?
                rr.log(
                    "go2/head_camera", rr.VideoStream.from_fields(sample=bytes(packet))
                )

        else:
            rr.log("go2/head_camera", rr.Image(rgb))
