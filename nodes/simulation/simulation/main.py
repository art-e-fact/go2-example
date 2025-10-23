"""TODO: Add docstring."""

import pyarrow as pa
from dora import Node


from enum import Enum

from simulation.scene_config import Scene
from simulation.check_nvidia_driver import check_nvidia_driver


class ControlMode(str, Enum):
    """Control mode for the simulation."""

    direct_inference = "direct_inference"
    dds_api = "dds_api"


def simulation():
    node = Node()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.physx.bundle")

    from simulation.go2_scene import EnvironmentRunner, simulate_unitree_sdk


    runner = EnvironmentRunner(simulation_app)
    runner.initialize()

    while runner.simulation_app.is_running():
        runner.step()

        # Consume all buffered events and then continue the simulation
        event = node.next(timeout=0.01)

        if event is None:
            continue

        # Skip event stream timeout errors (we use timeout=0.0 above)
        if event["type"] == "ERROR" and event["error"].startswith(
            "Timeout event stream error: Receiver timed out"
        ):
            continue

        if event["type"] == "INPUT":
            if event["id"] == "load_scene":
                print(f"Node received load_scene command: {event['value'][0].as_py()}")
                scene_info = event["value"][0].as_py()
                scene = Scene(scene_info['name'])
                runner.set_difficulty(scene_info["difficulty"])
                runner.load_scene(scene)

            elif event["id"] == "command_2d":
                command_2d = event["value"].to_pylist()
                x, y, yaw = command_2d
                runner.set_command(x, y, yaw)

            elif event["id"] == "pub_status_tick":
                node.send_output("rtf", pa.array([runner.get_rtf()]))
                node.send_output(
                    "waypoint_mission_complete",
                    pa.array([runner.waypoint_mission.is_complete()]),
                )
                node.send_output(
                    "waypoints",
                    pa.array(
                        [
                            {
                                "status": runner.waypoint_mission.get_waypoint_status(
                                    wp
                                ).value,
                                "position": wp.get_position()[0],
                            }
                            for wp in runner.waypoint_mission.waypoints
                        ]
                    ),
                )
                robot_pos, robot_quat = runner.get_robot_pose()
                node.send_output(
                    "robot_pose",
                    pa.array([{"position": robot_pos, "quaternion": robot_quat}]),
                )
                node.send_output(
                    "scene_info",
                    pa.array(
                        [
                            {
                                "name": runner.current_scene.name,
                                "difficulty": runner.difficulty,
                            }
                        ]
                    ),
                )

            elif event["id"] == "stop":
                print("Node received stop command, shutting down simulation.")
                runner.close()

        elif event["type"] == "STOP":
            print("Node received STOP event, shutting down simulation.")
            print(event)
            runner.close()

        elif event["type"] == "ERROR":
            print("Node received ERROR event:")
            print(event)


def main():
    check_nvidia_driver()
    simulation()


if __name__ == "__main__":
    main()
