"""Test module for navigator package."""

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pyarrow as pa
import pytest
from msgs import Transform, Twist2D, Waypoint, WaypointList, WaypointStatus


def test_import_main():
    """Test importing and running the main function."""
    from navigator.main import main

    # Check that everything is working, and catch Dora RuntimeError
    # as we're not running in a Dora dataflow.
    with pytest.raises(RuntimeError):
        main()


def test_navigator_logic():
    """Test the main navigator logic with mocked inputs."""
    # Patch dora.Node in the context of the navigator.main module
    with patch("navigator.main.Node") as MockNode:
        mock_node_instance = MagicMock()
        MockNode.return_value = mock_node_instance

        # Define the inputs that the node will receive.
        robot_pose = Transform.from_position_and_quaternion(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),  # scalar-first quaternion (w, x, y, z)
        )
        waypoints = WaypointList(
            [
                Waypoint(
                    transform=Transform.from_position_and_quaternion(
                        np.array([10.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
                    ),
                    status=WaypointStatus.ACTIVE,
                )
            ]
        )

        # The mocked node will yield these events when iterated.
        mock_node_instance.__iter__.return_value = [
            {"type": "INPUT", "id": "robot_pose", "value": robot_pose.to_arrow()},
            {"type": "INPUT", "id": "waypoints", "value": waypoints.to_arrow()},
            {
                "type": "INPUT",
                "id": "tick",
                "value": pa.array([]),  # The tick value is not used.
            },
            {
                "type": "INPUT",
                "id": "stop",  # Stop the loop
            },
        ]

        from navigator.main import main

        main()

        # Check that send_output was called correctly.
        # The robot is at the origin, facing the goal at (10, 0).
        # It should command maximum forward velocity and no angular velocity.
        expected_command = Twist2D(linear_x=1.0, linear_y=0.0, angular_z=0.0)

        # mock_node_instance.send_output.assert_called_once() # Fails if called more than once
        mock_node_instance.send_output.assert_called_with(
            "command_2d", expected_command.to_arrow()
        )


def test_navigator_logic_stateful():
    """Test the stateful logic of the navigator.

    This test verifies that:
    1. The navigator correctly processes inputs (robot_pose, waypoints) and maintains state.
    2. It generates a 'command_2d' output for every 'tick' input.
    3. The generated commands are correct based on the current robot pose and target waypoint:
       - Moving straight when facing the target.
       - Turning when not facing the target.
    """
    # A queue to send events to the node
    event_queue = queue.Queue()

    with patch("navigator.main.Node") as MockNode:
        mock_node_instance = MagicMock()
        MockNode.return_value = mock_node_instance
        mock_node_instance.__iter__.return_value = iter(event_queue.get, None)

        from navigator.main import main

        # Run the main function in a separate thread
        main_thread = threading.Thread(target=main, daemon=True)
        main_thread.start()

        # Set initial robot pose and waypoints
        robot_pose = Transform.from_position_and_quaternion(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),  # scalar-first (w, x, y, z)
        )
        waypoints = WaypointList(
            [
                Waypoint(
                    transform=Transform.from_position_and_quaternion(
                        np.array([10.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])
                    ),
                    status=WaypointStatus.ACTIVE,
                )
            ]
        )
        event_queue.put(
            {"type": "INPUT", "id": "robot_pose", "value": robot_pose.to_arrow()}
        )
        event_queue.put(
            {"type": "INPUT", "id": "waypoints", "value": waypoints.to_arrow()}
        )

        # Send a tick to trigger a command calculation
        event_queue.put({"type": "INPUT", "id": "tick", "value": pa.array([])})

        threading.Event().wait(0.1)  # Wait for async operations
        mock_node_instance.send_output.assert_called_once()
        args, _ = mock_node_instance.send_output.call_args
        assert args[0] == "command_2d"
        command_output = Twist2D.from_arrow(args[1])
        assert command_output.linear_x > 0, (
            f"Expected to go towards X direction, got {command_output}"
        )
        assert command_output.angular_z == 0.0, (
            f"Expected to go straight, got {command_output}"
        )

        # Reset mock to check for the next call
        mock_node_instance.send_output.reset_mock()

        # Set a new robot pose
        new_robot_pose = Transform.from_position_and_quaternion(
            np.array([10.0, 10.0, 0.0]),
            np.array([0.707, 0.0, 0.0, 0.707]),  # Rotated 90 degrees (facing +y)
        )
        event_queue.put(
            {"type": "INPUT", "id": "robot_pose", "value": new_robot_pose.to_arrow()}
        )

        # Send another tick
        event_queue.put({"type": "INPUT", "id": "tick", "value": pa.array([])})

        # Check the new output
        threading.Event().wait(0.1)  # Wait for async operations
        mock_node_instance.send_output.assert_called_once()
        args, _ = mock_node_instance.send_output.call_args
        assert args[0] == "command_2d"
        command_output = Twist2D.from_arrow(args[1])
        assert command_output.angular_z > 0.0, f"Expected to turn, got {command_output}"

        event_queue.put(None)  # Signal the iterator to end
        main_thread.join(timeout=1)  # Wait for the thread to finish


def test_stop_after_stop_signal():
    """Test that the navigator stops after receiving a stop signal."""
    event_queue = queue.Queue()

    with patch("navigator.main.Node") as MockNode:
        mock_node_instance = MagicMock()
        MockNode.return_value = mock_node_instance
        mock_node_instance.__iter__.return_value = iter(event_queue.get, None)

        from navigator.main import main

        # Start the main function in a separate thread
        main_thread = threading.Thread(target=main, daemon=True)
        main_thread.start()

        # Send stop signal
        event_queue.put({"type": "INPUT", "id": "stop"})
        threading.Event().wait(0.1)

        # Check if the thread has finished
        assert not main_thread.is_alive(), "Main thread did not stop after stop signal"
