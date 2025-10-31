from enum import Enum
from pyarrow_message import ArrowMessage
from dataclasses import dataclass, asdict
import pyarrow as pa
import numpy as np


@dataclass
class SceneInfo(ArrowMessage):
    name: str
    difficulty: float


@dataclass
class Timestamp(ArrowMessage):
    seconds: int
    nanoseconds: int

    @classmethod
    def from_float_seconds(cls, float_seconds: float) -> "Timestamp":
        seconds = int(float_seconds)
        nanoseconds = int((float_seconds - seconds) * 1e9)
        return cls(seconds=seconds, nanoseconds=nanoseconds)

    @classmethod
    def now(cls) -> "Timestamp":
        import time

        return cls.from_float_seconds(time.time())
    
    @property
    def float_seconds(self) -> float:
        return self.seconds + self.nanoseconds / 1e9


@dataclass
class Transform(ArrowMessage):
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    @classmethod
    def from_position_and_quaternion(
        cls, position: list[float], quaternion: list[float]
    ) -> "Transform":
        return cls(
            x=position[0],
            y=position[1],
            z=position[2],
            qx=quaternion[0],
            qy=quaternion[1],
            qz=quaternion[2],
            qw=quaternion[3],
        )

    @property
    def position(self) -> list[float]:
        return [self.x, self.y, self.z]

    @property
    def quaternion(self) -> list[float]:
        return [self.qx, self.qy, self.qz, self.qw]


@dataclass
class Twist2D(ArrowMessage):
    linear_x: float
    linear_y: float
    angular_z: float

@dataclass
class Observations(ArrowMessage):
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    gravity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    height_scan: np.ndarray

@dataclass
class JointCommands(ArrowMessage):
    positions: np.ndarray



class WaypointStatus(ArrowMessage, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"


@dataclass
class Waypoint(ArrowMessage):
    status: WaypointStatus
    transform: Transform


@dataclass
class WaypointList:
    waypoints: list[Waypoint]

    # TODO: Create a PR to pyarrow_message to support lists of ArrowMessages
    def to_arrow(self):
        waypoint_dicts = [
            {"status": wp.status.value, "transform": asdict(wp.transform)}
            for wp in self.waypoints
        ]
        return pa.array(waypoint_dicts)

    @classmethod
    def from_arrow(cls, data: pa.Array) -> "WaypointList":
        waypoint_dicts = data.to_pylist()
        waypoints = [
            Waypoint(
                status=WaypointStatus(wp_dict["status"]),
                transform=Transform(**wp_dict["transform"]),
            )
            for wp_dict in waypoint_dicts
        ]
        return cls(waypoints=waypoints)


if __name__ == "__main__":
    ts = Timestamp.now()
    print(f"Current timestamp: {ts.seconds}s {ts.nanoseconds}ns")

    # TODO: Add these tests to a proper test suite
    wp1 = Waypoint(
        status=WaypointStatus.ACTIVE,
        transform=Transform.from_position_and_quaternion(
            [1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]
        ),
    )
    wp2 = Waypoint(
        status=WaypointStatus.INACTIVE,
        transform=Transform.from_position_and_quaternion(
            [4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]
        ),
    )

    wpl = WaypointList(waypoints=[wp1, wp2])
    print(f"Waypoint List: {wpl}")
    wpl_arrow = wpl.to_arrow()
    wpl_back = WaypointList.from_arrow(wpl_arrow)
    print(f"Waypoint List (from Arrow): {wpl_back}")
    assert wpl == wpl_back
