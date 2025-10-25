"""Dora node that test waypoint mission completions using the robot pose."""

from dora import Node
import pytest
import pyarrow as pa
from posetree import Transform
import time

from nodes.tester.tester.transforms import Transforms


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
    transforms = Transforms()
    node.send_output(
        "load_scene", pa.array([{"name": scene, "difficulty": difficulty}])
    )

    waypoint_list: list[str] = []
    next_waypoint_index = 0
    
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "waypoints":
            waypoints: list[str | True] = event["value"].to_pylist()
            # Bail if waypoints are empty
            if not waypoints:
                continue
            # Initialize checklist
            for wp in waypoints:
                waypoint_frame = f"waypoint_{waypoints.index(wp)}"
                if waypoint_frame not in waypoint_list:
                    waypoint_list.append(waypoint_frame)
    
                transforms.add_transform(
                    Transform.from_position_and_quaternion(
                        wp["position"], wp["quaternion"]
                    ),
                    int(time.time()),
                    parent_frame="world",
                    child_frame=waypoint_frame,
                )

        elif event["type"] == "INPUT" and event["id"] == "robot_pose":
            # Wait for waypoints to be registered
            if len(waypoint_list) == 0:
                continue

            pose = event["value"].to_pylist()[0]
            transform = Transform.from_position_and_quaternion(
                pose["position"], pose["quaternion"]
            )
            timestamp = int(time.time())
            transforms.add_transform(
                transform,
                timestamp,
                parent_frame="world",
                child_frame="robot",
            )
            
            distance_threshold = 0.5
            goal_frame = waypoint_list[next_waypoint_index]
            distance = transforms.distance_to(
                from_frame="robot",
                to_frame=goal_frame,
                timestamp=timestamp,
            )
            if distance < distance_threshold:
                print(f"Reached waypoint {next_waypoint_index}")

                if next_waypoint_index < len(waypoint_list) - 1:
                    next_waypoint_index += 1
                else:
                    print("All waypoints completed!")
                    break
        
