"""Policy controller node."""

from pathlib import Path

import pyarrow as pa
from dora import Node

import msgs
from policy_controller.policy import Policy


def main():
    """Receive observations and twist commands and output joint commands."""
    node = Node()

    policy = Policy(
        model_path=Path(__file__).parent / "policy" / "policy.pt",
        config_path=Path(__file__).parent / "policy" / "env.yaml",
    )
    control_timestep = policy.config.dt * policy.config.decimation

    last_observations = None
    last_command_2d = None
    last_step_time = 0.0
    last_time = None
    last_commands = None

    def try_step():
        nonlocal last_observations, last_command_2d, last_step_time, last_time
        if last_observations is None or last_command_2d is None or last_time is None:
            return

        # If the world reset
        if last_time < last_step_time:
            policy.reset()
            last_observations = None
            last_command_2d = None
            last_step_time = 0.0

        if last_time - last_step_time >= control_timestep - 1e-6:
            last_step_time = last_time

            joint_targets = policy.forward(
                observation=last_observations,
                command=last_command_2d,
            )
            node.send_output(
                "joint_commands",
                msgs.JointCommands(positions=joint_targets).to_arrow(),
            )
        elif last_commands is not None:
            # Resend last commands
            node.send_output(
                "joint_commands",
                last_commands,
            )

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "command_2d":
                last_command_2d = msgs.Twist2D.from_arrow(event["value"])
            elif event["id"] == "observations":
                last_observations = msgs.Observations.from_arrow(event["value"])
                try_step()
            elif event["id"] == "clock":
                last_time = msgs.Timestamp.from_arrow(event["value"]).float_seconds


if __name__ == "__main__":
    main()
