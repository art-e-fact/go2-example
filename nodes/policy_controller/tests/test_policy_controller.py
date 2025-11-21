"""Test module for policy_controller package."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from msgs import JointCommands, Observations, Timestamp, Twist2D


def test_import_main():
    """Test importing and running the main function."""
    from policy_controller.main import main

    # Check that everything is working, and catch Dora RuntimeError
    # as we're not running in a Dora dataflow.
    with pytest.raises(RuntimeError):
        main()


def test_generates_commands():
    """Check if the policy runs and generates joint commands."""
    with patch("policy_controller.main.Node") as MockNode:
        mock_node_instance = MagicMock()
        MockNode.return_value = mock_node_instance

        # Create mock inputs
        command_2d = Twist2D(linear_x=0.5, linear_y=0.0, angular_z=0.0)
        observations = Observations(
            lin_vel=np.zeros(3),
            ang_vel=np.zeros(3),
            gravity=np.array([0.0, 0.0, -9.81]),
            joint_positions=np.zeros(12),
            joint_velocities=np.zeros(12),
            height_scan=np.zeros(154),
        )
        clock = Timestamp.now()

        # The mocked node will yield these events when iterated.
        mock_node_instance.__iter__.return_value = [
            {"type": "INPUT", "id": "command_2d", "value": command_2d.to_arrow()},
            {"type": "INPUT", "id": "clock", "value": clock.to_arrow()},
            {
                "type": "INPUT",
                "id": "observations",
                "value": observations.to_arrow(),
            },
        ]

        from policy_controller.main import main

        main()

        # Check that send_output was called with joint_commands
        mock_node_instance.send_output.assert_called()
        args, _ = mock_node_instance.send_output.call_args
        assert args[0] == "joint_commands"
        joint_commands = JointCommands.from_arrow(args[1])
        assert joint_commands.positions is not None
        assert len(joint_commands.positions) == 12
