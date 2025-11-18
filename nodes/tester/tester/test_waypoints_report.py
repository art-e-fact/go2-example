"""Dora node that test waypoint mission completions."""

import pytest
import msgs


@pytest.mark.clock_timeout(50)
def test_receives_scene_info_on_startup(node):
    """Test that the node receives scene info on startup."""
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "scene_info":
            scene_info = msgs.SceneInfo.from_arrow(event["value"])
            assert scene_info.name is not None
            assert scene_info.difficulty is not None
            return


@pytest.mark.parametrize("difficulty", [0.1, 0.7, 1.1])
@pytest.mark.clock_timeout(30)
def test_completes_waypoint_mission_with_variable_height_steps(node, difficulty: float):
    """Test that the waypoint mission completes successfully.

    The pyramid steps height is configured via difficulty.
    """
    run_waypoint_mission_test(node, scene="generated_pyramid", difficulty=difficulty)


@pytest.mark.parametrize("scene", ["rail_blocks", "stone_stairs", "excavator"])
@pytest.mark.clock_timeout(50)
def test_completes_waypoint_mission_in_photo_realistic_env(node, scene: str):
    """Test that the waypoint mission completes successfully."""
    run_waypoint_mission_test(node, scene, difficulty=1.0)


def run_waypoint_mission_test(node, scene: str, difficulty: float):
    """Run a waypoint mission test with the given scene and difficulty."""
    print(
        f"Starting waypoint mission test in scene '{scene}' with difficulty {difficulty}"
    )
    node.send_output(
        "load_scene", msgs.SceneInfo(name=scene, difficulty=difficulty).to_arrow()
    )

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "waypoints":
                waypoints = msgs.WaypointList.from_arrow(event["value"]).waypoints
                if len(waypoints) == 0:
                    continue
                if all(wp.status == msgs.WaypointStatus.COMPLETED for wp in waypoints):
                    return
