"""TODO: Add docstring."""

import pyarrow as pa
import numpy as np
from dora import Node
from navigator.compute_command import compute_command
import msgs


def main():
    """TODO: Add docstring."""
    node = Node()

    print("Navigator node started.")

    last_robot_pose = None
    last_waypoints = None

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "tick":
                if last_robot_pose is None or last_waypoints is None:
                    continue

                next_waypoint = None
                for wp in last_waypoints:
                    if wp.status == msgs.WaypointStatus.ACTIVE:
                        next_waypoint = wp
                        break

                if next_waypoint is None:
                    node.send_output(
                        "command_2d",
                        msgs.Twist2D(
                            linear_x=0.0, linear_y=0.0, angular_z=0.0
                        ).to_arrow(),
                    )
                    continue

                command = compute_command(
                    robot_pose=last_robot_pose,
                    goal_position=np.array(next_waypoint.transform.position),
                )

                node.send_output(
                    "command_2d",
                    msgs.Twist2D(
                        linear_x=command[0], linear_y=command[1], angular_z=command[2]
                    ).to_arrow(),
                )

            elif event["id"] == "waypoints":
                last_waypoints = msgs.WaypointList.from_arrow(event["value"]).waypoints

            elif event["id"] == "robot_pose":
                last_robot_pose = msgs.Transform.from_arrow(event["value"])

            elif event["id"] == "my_input_id":
                # Warning: Make sure to add my_output_id and my_input_id within the dataflow.
                node.send_output(
                    output_id="my_output_id",
                    data=pa.array([1, 2, 3]),
                    metadata={},
                )

            elif event["id"] == "stop":
                print("Navigator node received stop command, shutting down.")
                break


if __name__ == "__main__":
    main()
