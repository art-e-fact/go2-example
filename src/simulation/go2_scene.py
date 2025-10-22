import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import carb.input
import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim import SimulationApp
from isaacsim.core.api import World
from isaacsim.core.utils import stage as stage_utils
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, Sdf, UsdGeom

from simulation.auto_pilot import AutoPilot
from simulation.camera_manager import CameraManager
from simulation.devices.gamepad import Se2Gamepad
from simulation.devices.keyboard import Se2Keyboard
from simulation.environments.pyramid import create_stepped_pyramid
from simulation.environments.rails import create_rails
from simulation.follow_camera import FollowCamera
from simulation.go2_robot import Go2Policy
from simulation.input_listener import InputListener
from simulation.rerun_logger import RerunLogger
from simulation.rtf_calculator import RtfCalculator
from simulation.scene_config import Scene
from simulation.steady_rate import SteadyRate
from simulation.waypoint_mission import WaypointMission

SCENE_ROOT = "/Scene"


def add_reference(
    asset_path: str,
    translation: Optional[Gf.Vec3d] = None,
    rotation: Optional[Gf.Rotation] = None,
    path: Optional[str] = None,
):
    if not asset_path.startswith("https://"):
        asset_path = str(Path(__file__).parent / asset_path)
        if not Path(asset_path).exists():
            raise FileNotFoundError(
                f"Asset not found: {asset_path}\nDid you run `./scripts/update_third_party_dependencies.sh`?"
            )
    if path:
        path = Sdf.Path(path)
    else:
        path = Sdf.Path(SCENE_ROOT).AppendChild("Env")
    prim = stage_utils.add_reference_to_stage(usd_path=asset_path, prim_path=path)

    if translation is not None or rotation is not None:
        xform = UsdGeom.Xformable(prim)
        transform = xform.AddTransformOp()
        mat = Gf.Matrix4d()
        if translation is not None:
            mat.SetTranslateOnly(translation)
        if rotation is not None:
            mat.SetRotateOnly(rotation)

        transform.Set(mat)


def set_attr(prim, attr_name, value):
    if type(prim) is str:
        prim = get_prim_at_path(prim)

    phys_attr = prim.GetAttribute(attr_name)
    if phys_attr:
        success = phys_attr.Set(value)
        if not success:
            print(f"Failed to set attribute {attr_name} in {prim.GetPath()}")
    else:
        print(f"Attribute {attr_name} not found in {prim.GetPath()}")


@dataclass
class DemoSceneConfig:
    scene_name: str
    follow_camera_location: Tuple[float, float, float] = field(
        default_factory=lambda: (5.0, 5.0, 5.0)
    )
    robot_position: Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]
    ] = None  # (x,y,z), (w,x,y,z) quaternion


def add_assets_to_world(scene: Scene, difficulty: float = 1.0) -> DemoSceneConfig:
    config = DemoSceneConfig(scene_name=scene.value.title())

    if scene == Scene.grid:
        add_reference("environments/grid.usda")
    elif scene == Scene.generated_rail:
        add_reference("environments/grid.usda")
        create_rails(prim_path=SCENE_ROOT + "/Rails")
    elif scene == Scene.generated_pyramid:
        add_reference("environments/grid.usda")
        create_stepped_pyramid(
            prim_path=SCENE_ROOT + "/Pyramid",
            position=(2.0, 0.0, -0.45),
            step_height=0.08 * difficulty,
        )
    elif scene == Scene.hospital_staircase:
        add_reference("../../assets/jouer/jouer.usda")
        config.follow_camera_location = (-0.26, -5.46, 1.31)
        config.robot_position = (
            (-1.132995391295986, -3.9926660542312136, -1.3211653993039363e-14),
            (0.7108333798924511, 0, 0, 0.7033604382041077),
        )
    elif scene == Scene.rail_blocks:
        add_reference("../../assets/rail_blocks/rail_blocks.usd")
        config.robot_position = (
            (-9.160084778124597, 3.7498660414059666, -4.440892098500626e-16),
            (0.9777604847551223, 0, 0, -0.2097246634314338),
        )
    elif scene == Scene.stone_stairs:
        add_reference("../../assets/stone_stairs/stone_stairs_f.usd")
        config.follow_camera_location = (
            1.922574758064275,
            -4.040539873363723,
            1.701546649598252,
        )
    elif scene == Scene.excavator:
        add_reference("../../assets/excavator_scan/excavator.usd")
        config.follow_camera_location = (
            2.345979532852949,
            2.4994179277846036,
            1.847139190251859,
        )
        config.robot_position = (
            (0, 0, 0),
            (0.45157244431278176, 0, 0, 0.892234457716905),
        )

    else:
        raise ValueError(f"Scene {scene} not recognized")

    return config


class EnvironmentRunner:
    def __init__(
        self,
        simulation_app: SimulationApp,
        first_scene: Scene = Scene.grid,
        use_rerun: bool = False,
        use_video_stream: bool = False,
        use_auto_pilot: bool = False,
        rrd_path: str = None,
        difficulty: float = 0.5,
    ):
        self.scene_circulation = [
            Scene.hospital_staircase,
            Scene.stone_stairs,
            Scene.rail_blocks,
            Scene.excavator,
        ]
        self.simulation_app = simulation_app
        self.current_scene = first_scene
        self.use_rerun = use_rerun
        self.use_video_stream = use_video_stream
        self.use_auto_pilot = use_auto_pilot
        self.rrd_path = rrd_path
        self.difficulty = difficulty
        self.robot_path = "/World/Go2"
        self.base_command = np.zeros(3)
        self.rendering_dt = 1 / 25.0
        self.steady_rate = SteadyRate(rate_hz=1.0 / self.rendering_dt)
        self._switch_keyboard = InputListener(
            carb.input.KeyboardInput.TAB, self.load_next_scene
        )
        self._switch_gamepad = InputListener(
            carb.input.GamepadInput.MENU2, self.load_next_scene
        )
        self._reset_keyboard = InputListener(
            carb.input.KeyboardInput.BACKSPACE, self.reload_scene
        )
        self._reset_gamepad = InputListener(
            carb.input.GamepadInput.MENU1, self.reload_scene
        )
        self._needed_reset = False
        self._is_initializing = True
        self._on_its_back_since = None
        self._on_its_back_since_threshold = 2.0  # seconds
        self._rtf_calculator = RtfCalculator(
            window_size=200, update_interval=100
        )  # 1-second window for 200Hz, update every 100 steps
        self.log_rtf = False

    def initialize(self):
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1 / 200,
            rendering_dt=self.rendering_dt,
        )
        self.world.reset()

        self.go2 = Go2Policy(
            prim_path=self.robot_path,
            name="Go2",
        )
        if self.use_rerun:
            self.rerun_logger = RerunLogger(self.robot_path, self.use_video_stream)

        self.teleop_keyboard = Se2Keyboard()
        self.teleop_gamepad = Se2Gamepad()

        self.follow_camera = FollowCamera(target_prim_path=self.robot_path)
        self.follow_camera.initialize()

        self.camera_manager = CameraManager(self.follow_camera.camera)

        self.initialize_scene()

        self.world.add_physics_callback(
            "physics_step", callback_fn=self.on_physics_step
        )
        # signal.signal(signal.SIGTERM, self._handle_shutdown)
        # signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        print(f"Received shutdown signal: {signum}. Shutting down gracefully.")
        # This will cause the main loop in `run()` to exit.
        self.close()

    def initialize_scene(self):
        self._is_initializing = True
        demo_config = add_assets_to_world(self.current_scene, self.difficulty)

        self.world.reset()
        self.go2.reset()
        self.go2.initialize()

        if demo_config.robot_position is not None:
            # Set the initial robot position in every possible way. I'm confuesed how the reset prcedure works in Isaac,
            # but doing it this way seems to work, so we can figure that out later.
            position, orientation = demo_config.robot_position
            self.go2.robot.set_world_pose(
                position=np.array(position), orientation=np.array(orientation)
            )

            self.go2.robot.set_default_state(
                position=np.array(position), orientation=np.array(orientation)
            )
            set_attr(self.go2.robot.prim, "xformOp:translate", Gf.Vec3f(*position))
            set_attr(self.go2.robot.prim, "xformOp:orient", Gf.Quatd(*orientation))

        # Add some steps to stabilize the robot (TODO: figure out how to do this properly)
        self.world.step(render=True)
        self.world.step(render=True)
        self.world.step(render=True)

        self.follow_camera.camera_location = np.array(
            demo_config.follow_camera_location
        )
        self.follow_camera.reset()

        self.waypoint_mission = WaypointMission()
        self.waypoint_mission.initialize()
        if self.use_rerun:
            self.rerun_logger.initialize(self.rrd_path)
            self.rerun_logger.set_scene_name(demo_config.scene_name)
            self.rerun_logger.link_waypoint_mission(self.waypoint_mission)
            self.rerun_logger.link_rtf_calculator(self._rtf_calculator)
        self._rtf_calculator.reset()

        self._is_initializing = False

        if self.use_auto_pilot:
            self.auto_pilot = AutoPilot(self.waypoint_mission)

        self.camera_manager.initialize()
        self.camera_manager.link_waypoint_mission(self.waypoint_mission)

    def load_next_scene(self):
        print("Loading next scene...")
        current_index = (
            self.scene_circulation.index(self.current_scene)
            if self.current_scene in self.scene_circulation
            else -1
        )
        next_index = (current_index + 1) % len(self.scene_circulation)

        self.load_scene(self.scene_circulation[next_index])

    def load_scene(self, scene: Scene):
        print(f"Loading scene {scene}...")
        self.current_scene = scene

        self.world.stop()

        omni.kit.commands.execute("DeletePrims", paths=[SCENE_ROOT], destructive=False)

        self.initialize_scene()

    def reload_scene(self):
        self.load_scene(self.current_scene)

    def on_physics_step(self, step_size) -> None:
        rtf = self._rtf_calculator.step(step_size)
        if rtf is not None and self.log_rtf:
            print(f"Real-Time Factor (RTF): {rtf:.2f}")

        if not self.go2.robot.handles_initialized:
            print("Robot not initialized yet")
            return

        if not self._is_initializing:
            try:
                self.go2.forward(step_size, self.base_command)
            except ValueError as e:
                print(f"Error in policy forward: {e}")
                self._needed_reset = True

    def check_if_robot_is_on_its_back(self):
        if not self.go2.is_on_back():
            self._on_its_back_since = None
            return

        if self._on_its_back_since is None:
            self._on_its_back_since = time.time()

        elif time.time() - self._on_its_back_since > self._on_its_back_since_threshold:
            print("Robot has been on its back for too long, resetting...")
            self._needed_reset = True
            self._on_its_back_since = None

    def run(self):
        while self.simulation_app.is_running():
            self.step()
        print("Simulation app has exited main loop, cleaning up...")
        self.close()
        print("Environment runner cleanup complete.")

    def step(self):
        self.check_if_robot_is_on_its_back()

        if self._needed_reset:
            print("Resetting environment...")
            self._needed_reset = False
            self.reload_scene()
            return

        self.world.step(render=True)

        self.follow_camera.update()
        self.waypoint_mission.update()
        if self.use_rerun:
            self.rerun_logger.log()

        self.camera_manager.capture_frames()

        if self.world.is_playing():
            # Use the device that has input
            if self.use_auto_pilot and self.auto_pilot is not None:
                self.base_command = self.auto_pilot.advance()
            else:
                self.base_command = self.teleop_keyboard.advance()
                if np.all(self.base_command == 0):
                    self.base_command = self.teleop_gamepad.advance()

        # Prevent the RTF from going above 1.0 (faster than real-time)
        self.steady_rate.sleep()
        

    def close(self):
        self.camera_manager.close()
        self.simulation_app.close()


# TODO use the EnvironmentRunner for this too

_first_step = True
_reset_needed = False


def simulate_unitree_sdk(simulation_app: SimulationApp, scene: Scene):
    from dds.dds_master import dds_manager
    from dds.go2_robot_dds import Go2RobotDDS

    publish_names = []
    subscribe_names = []
    go2_robot_dds = Go2RobotDDS()
    dds_manager.register_object("go2", go2_robot_dds)
    publish_names.append("go2")
    subscribe_names.append("go2")
    dds_manager.start_publishing(publish_names)
    dds_manager.start_subscribing(subscribe_names)

    # initialize robot on first step, run robot advance
    def on_physics_step(step_size) -> None:
        global _first_step
        global _reset_needed
        if _first_step:
            go2.initialize()
            _first_step = False
        elif _reset_needed:
            my_world.reset(True)
            _reset_needed = False
            _first_step = True
        else:
            go2.read_targets_from_dds(go2_robot_dds)
            go2.write_robot_dds_state(go2_robot_dds)

    # spawn world
    my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 500, rendering_dt=1 / 50)

    add_assets_to_world(scene)

    # spawn robot
    go2 = Go2Policy(
        prim_path="/World/Go2",
        name="Go2",
        position=np.array([0, 0, 0.0]),
    )
    my_world.reset()
    my_world.add_physics_callback("physics_step", callback_fn=on_physics_step)

    dt = 1 / 60.0
    while simulation_app.is_running():
        start_time = time.time()
        my_world.step(render=True)
        if my_world.is_stopped():
            global _reset_needed
            _reset_needed = True

    simulation_app.close()
