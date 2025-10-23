"""TODO: Add docstring."""

import pyarrow as pa
import numpy as np
from dora import Node
from navigator.compute_command import compute_command


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
                    if wp["status"] == "active":
                        next_waypoint = wp
                        break

                if next_waypoint is None:
                    print("No active waypoint found, stopping robot.")
                    node.send_output("command_2d", pa.array([0, 0, 0]))
                    continue

                command = compute_command(
                    robot_position=np.array(last_robot_pose["position"]),
                    robot_quaternion=np.array(last_robot_pose["quaternion"]),
                    goal_position=np.array(next_waypoint["position"]),
                )

                node.send_output("command_2d", pa.array(command))

            elif event["id"] == "waypoints":
                last_waypoints = event["value"].to_pylist()
            
            elif event["id"] == "robot_pose":
                last_robot_pose = event["value"][0].as_py()
                

            elif event["id"] == "my_input_id":
                # Warning: Make sure to add my_output_id and my_input_id within the dataflow.
                node.send_output(
                    output_id="my_output_id", data=pa.array([1, 2, 3]), metadata={},
                )


if __name__ == "__main__":
    main()
