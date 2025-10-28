"""TODO: Add docstring."""

from pathlib import Path
import msgs
import pyarrow as pa
from dora import Node

from policy_controller.policy import Policy


def main():
    """TODO: Add docstring."""
    node = Node()

    policy = Policy(
        model_path=Path(__file__).parent / "policy" / "policy.pt",
        config_path=Path(__file__).parent / "policy" / "env.yaml",
    )
    control_timestep = policy.config.dt * policy.config.decimation

    last_observations = None
    last_command_2d = None
    last_step_time = 0.0

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "command_2d":
                last_command_2d = msgs.Twist2D.from_arrow(event["value"])
            elif event["id"] == "observations":
                last_observations = msgs.Observations.from_arrow(event["value"])
            elif event["id"] == "clock":
                time = msgs.Timestamp.from_arrow(event["value"]).float_seconds
                # If the world reset
                if time < last_step_time:
                    policy.reset()
                    last_observations = None
                    last_command_2d = None
                    last_step_time = 0.0
                    
                if time - last_step_time >= control_timestep:
                    last_step_time = time

                    if last_observations is None or last_command_2d is None:
                        continue

                    joint_targets = policy.forward(
                        observation=last_observations,
                        command=last_command_2d,
                    )
                    node.send_output(
                        "joint_commands",
                        msgs.JointCommands(positions=joint_targets).to_arrow(),
                    )


if __name__ == "__main__":
    main()
