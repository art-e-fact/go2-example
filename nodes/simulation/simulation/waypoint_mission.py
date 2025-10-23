import re
import time
from enum import Enum
from typing import List, Tuple, Union

import numpy as np
import omni
import omni.usd
from isaacsim.core.prims import SingleXFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux


def find_prims_by_name_pattern(pattern: str) -> List[Usd.Prim]:
    """
    Finds all prims on the stage whose names match the given regex pattern.

    Args:
        pattern: A regular expression to match against prim names.

    Returns:
        A list of Usd.Prim objects that match the pattern.
    """
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("Error: Could not get USD stage.")
        return []

    regex = re.compile(pattern)
    matching_prims = []

    # stage.Traverse() iterates over all prims in the scene
    for prim in stage.Traverse():
        if regex.match(prim.GetName()):
            matching_prims.append(prim)

    return matching_prims


class WaypointStatus(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"


class Waypoint:
    def __init__(self, prim_path: Union[str, Sdf.Path]):
        stage = omni.usd.get_context().get_stage()
        self.prim = stage.GetPrimAtPath(prim_path)

        # Clean up any existing children
        deactivate_all_children(prim_path)

        # Create the disk light
        light_path = self.prim.GetPath().AppendPath("DiskLight")
        self.light_prim = create_disk_light(
            prim_path=light_path,
            position=Gf.Vec3f(0.0, 0.0, 1.0),
            radius=0.2,
            intensity=50000.0,
            color=Gf.Vec3f(0.0, 0.0, 1.0),  # A cool fill light
        )

        # Create the indicator sphere
        self.indicator_path = self.prim.GetPath().AppendPath("Indicator")
        omni.kit.commands.execute(
            "CreateMeshPrim",
            prim_type="Sphere",
            select_new_prim=False,
            half_scale=5.0,
            prim_path=self.indicator_path,
        )

        self.waving_progress = 0.0
        self.waving_speed = 0.03
        self.target_height = 0.35

    def get_position(self) -> Tuple[np.ndarray, np.ndarray]:
        # Waypoints don't move, so memoize the position on first call
        if not hasattr(self, "_position"):
            xform = SingleXFormPrim(str(self.prim.GetPath()))
            self._position = xform.get_world_pose()
        return self._position

    def set_inactive(self):
        stage = omni.usd.get_context().get_stage()

        self.light_prim.GetIntensityAttr().Set(0.0)
        mesh = UsdGeom.Mesh.Get(stage, self.indicator_path)
        self.target_height = 0.4
        mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.3, 0.3, 0.3)])

    def set_active(self):
        stage = omni.usd.get_context().get_stage()

        self.light_prim.GetIntensityAttr().Set(1000000.0)
        mesh = UsdGeom.Mesh.Get(stage, self.indicator_path)
        self.target_height = 0.3
        mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)])

    def set_completed(self):
        stage = omni.usd.get_context().get_stage()

        self.light_prim.GetIntensityAttr().Set(10000.0)
        self.light_prim.GetColorAttr().Set(Gf.Vec3f(0.0, 1.0, 0.0))
        mesh = UsdGeom.Mesh.Get(stage, self.indicator_path)
        self.target_height = 0.8
        mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])

    def update(self):
        xform = SingleXFormPrim(str(self.indicator_path))
        current_pos, orientation = xform.get_local_pose()
        self.waving_progress += self.waving_speed
        wave_offset = 0.01 * np.sin(self.waving_progress)
        target_z = self.target_height + wave_offset
        diff = target_z - current_pos[2]
        new_z = current_pos[2] + 0.1 * diff
        xform.set_local_pose([current_pos[0], current_pos[1], new_z], orientation)


class WaypointMission:
    def __init__(self, robot_path: str = "/World/Go2/base"):
        self.waypoints: list[Waypoint] = []
        self.current_index = 0
        self.robot_path = robot_path
        self.distance_threshold = 0.7  # meters
        self.mission_start_time = None
        self.waypoint_completed_times = []
        self.mission_end_time = None

    def initialize(self):
        # Find all prims with names matching "Waypoint_*"
        waypoint_prims = find_prims_by_name_pattern(r"Waypoint_.*")
        waypoint_prims.sort(key=lambda p: p.GetName())

        for waypoint_prim in waypoint_prims:
            waypoint = Waypoint(waypoint_prim.GetPath())
            self.waypoints.append(waypoint)

        self.current_index = 0
        self.mission_start_time = time.time()
        self.waypoint_completed_times = []
        self.mission_end_time = None
        self.update_waypoint_states()

    def update_waypoint_states(self):
        for wp in self.waypoints:
            status = self.get_waypoint_status(wp)
            if status == WaypointStatus.INACTIVE:
                wp.set_inactive()
            elif status == WaypointStatus.ACTIVE:
                wp.set_active()
            elif status == WaypointStatus.COMPLETED:
                wp.set_completed()

    def update(self):
        for wp in self.waypoints:
            wp.update()

        if not self.waypoints or self.is_complete():
            return

        if self.current_index >= len(self.waypoints):
            return

        robot_xform = SingleXFormPrim(str(self.robot_path))
        robot_pos, _ = robot_xform.get_world_pose()
        waypoint = self.waypoints[self.current_index]
        waypoint_pos, _ = waypoint.get_position()
        distance = np.linalg.norm(waypoint_pos - robot_pos)

        if distance < self.distance_threshold:
            print(
                f"Reached waypoint {self.current_index}, {len(self.waypoints) - self.current_index - 1} to go"
            )
            self.waypoint_completed_times.append(time.time())
            self.current_index += 1

            if self.current_index < len(self.waypoints):
                print(f"Next waypoint: {self.current_index}")
            else:
                print("All waypoints reached!")
                self.mission_end_time = time.time()

            self.update_waypoint_states()

    def get_waypoint_status(self, waypoint: Waypoint) -> WaypointStatus:
        index = self.waypoints.index(waypoint)
        if index < self.current_index:
            return WaypointStatus.COMPLETED
        elif index == self.current_index:
            return WaypointStatus.ACTIVE
        else:
            return WaypointStatus.INACTIVE

    def is_complete(self) -> bool:
        return self.mission_end_time is not None


def create_disk_light(
    prim_path: str,
    position: Gf.Vec3f,
    radius: float = 0.5,
    intensity: float = 30000.0,
    color: Gf.Vec3f = Gf.Vec3f(1.0, 1.0, 1.0),
):
    # Get the current USD stage
    stage = omni.usd.get_context().get_stage()

    # Create the DiskLight prim at the specified path
    light_prim = UsdLux.DiskLight.Define(stage, prim_path)
    shaping = UsdLux.ShapingAPI.Apply(light_prim.GetPrim())
    shaping.GetShapingConeAngleAttr().Set(22.0)

    if not light_prim:
        print(f"Error: Failed to create light at {prim_path}")
        return

    # Set the light's properties
    light_prim.GetRadiusAttr().Set(radius)
    light_prim.GetIntensityAttr().Set(intensity)
    light_prim.GetColorAttr().Set(color)

    # An alternative to intensity is exposure (logarithmic scale)
    # light_prim.GetExposureAttr().Set(10.0)

    # Move the light to the desired position
    # UsdGeom.XformCommonAPI is a standard way to handle transforms
    xform = SingleXFormPrim(str(prim_path))
    xform.set_local_pose(position, [1, 0, 0, 0])

    print(f"Successfully created disk light at {prim_path}")
    return light_prim


def deactivate_all_children(prim_path: Union[str, Sdf.Path]):
    """
    Removes all direct children of a prim at the given path.

    Args:
        prim_path: The path of the parent prim.
    """
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("Error: Could not get USD stage.")
        return

    parent_prim = stage.GetPrimAtPath(prim_path)
    if not parent_prim.IsValid():
        # If the prim doesn't exist, there's nothing to do.
        return

    # Iterate over a copy of the children list, as it will be modified during iteration.
    children_to_remove = list(parent_prim.GetChildren())
    print(f"Found {len(children_to_remove)} children to remove under {prim_path}.")

    for child_prim in children_to_remove:
        omni.kit.commands.execute(
            "ToggleActivePrims",
            stage_or_context=omni.usd.get_context().get_stage(),
            prim_paths=[child_prim.GetPath()],
            active=False,
        )
