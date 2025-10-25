"""Dora node that test waypoint mission completions."""

from typing import List, Optional
from dora import Node
import pytest
import pyarrow as pa
from transforms_py import PyRegistry
from posetree import CustomFramePoseTree, Transform, Pose
import time


class TransformsPoseTree(CustomFramePoseTree):
    """Pose tree that retrieves transforms from a PyRegistry."""

    def __init__(self, registry: PyRegistry):
        super().__init__()
        self._registry = registry

    def _get_transform(
        self, parent_frame: str, child_frame: str, timestamp: float = 0
    ) -> Transform:
        transform_data = self._registry.get_transform(
            parent_frame, child_frame, timestamp
        )
        if transform_data is None:
            raise KeyError(
                f"No transform found from {parent_frame} to {child_frame} at time {timestamp}"
            )
        tx, ty, tz, qx, qy, qz, qw, _ts, _parent, _child = transform_data
        return Transform.from_position_and_quaternion([tx, ty, tz], [qx, qy, qz, qw])

    def add_transform(
        self,
        transform: Transform,
        timestamp: float,
        parent_frame: str,
        child_frame: str,
    ):
        """Add a transform to the registry."""
        self._registry.add_transform(
            *transform.position, *transform.rotation.as_quat(), timestamp, parent_frame, child_frame
        )


@pytest.fixture(scope="session")
def node():
    """Create a Dora node for testing."""
    node = Node()
    yield node
    node.send_output("stop", pa.array([True]))


@pytest.mark.timeout(50)
def test_receives_scene_info_on_startup(node: Node):
    """Test that the node receives scene info on startup."""
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "scene_info":
            scene_info = event["value"].to_pylist()[0]
            assert "name" in scene_info
            assert "difficulty" in scene_info
            return


@pytest.mark.parametrize("difficulty", [0.1, 0.7, 1.1])
@pytest.mark.timeout(50)
def test_completes_waypoint_mission_with_variable_height_steps(
    node: Node, difficulty: float
):
    """Test that the waypoint mission completes successfully.

    The pyramid steps height is configured via difficulty.
    """
    run_waypoint_mission_test(node, scene="generated_pyramid", difficulty=difficulty)


@pytest.mark.parametrize(
    "scene", ["hospital_staircase", "rail_blocks", "stone_stairs", "excavator"]
)
@pytest.mark.timeout(50)
def test_completes_waypoint_mission_in_photo_realistic_env(node: Node, scene: str):
    """Test that the waypoint mission completes successfully."""
    run_waypoint_mission_test(node, scene, difficulty=1.0)


def run_waypoint_mission_test(node: Node, scene: str, difficulty: float):
    pose_tree = TransformsPoseTree()
    robot_pose = Pose.from_position_and_quaternion(
        [0, 0, 0], [0, 0, 0, 1], "odom", pose_tree
    )

    node.send_output(
        "load_scene", pa.array([{"name": scene, "difficulty": difficulty}])
    )

    waypoint_checklist = []

    # Wait for the waypoints positions
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "waypoints":
            waypoints: List[Pose | True] = event["value"].to_pylist()
            # Initialize checklist
            if len(waypoint_checklist) != len(waypoints):
                waypoint_checklist = [
                    Pose.from_position_and_quaternion(wp["position"], [0, 0, 0, 1], "world", pose_tree)
                    for wp in waypoints
                ]
                break

    # Check if the robot reaches all waypoints
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "robot_pose":
            position, orientation = event["value"].to_pylist()[0]
            transform = Transform.from_position_and_quaternion(position, orientation)

            timestamp = time.time()
            pose_tree.add_transform(
                transform,
                timestamp,
                parent_frame="world",
                child_frame="odom",
            )
            for i, wp in enumerate(waypoint_checklist):
                if wp is True:
                    continue
                if robot_pose.distance_to(wp, timestamp) < 0.5:
                    waypoint_checklist[i] = True
            if all(wp is True for wp in waypoint_checklist):
                print("All waypoints reached.")
                break
        
