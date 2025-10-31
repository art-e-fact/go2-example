"""TODO: Add docstring."""

from enum import Enum

import msgs
import pyarrow as pa
from dora import Node

from simulation.check_nvidia_driver import check_nvidia_driver
from simulation.scene_config import Scene
from simulation.simulation_time_output import SimulationTimeOutput


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

    # Publish simulation time at each physics step
    _simulation_time_output = SimulationTimeOutput(node, runner.world)

    # Publish observations at each physics step
    def on_physics_step(dt: float):
        observations = runner.go2.compute_observations()
        node.send_output("observations", observations.to_arrow())

    runner.world.add_physics_callback("observation_output", on_physics_step)

    while runner.simulation_app.is_running():
        # Consume all buffered events and then continue the simulation
        event = node.next(timeout=0.001)

        # The event buffer is empty, step the simulation
        if event["type"] == "ERROR" and event["error"].startswith(
            "Timeout event stream error: Receiver timed out"
        ):
            runner.step()
            continue

        if event["type"] == "INPUT":
            if event["id"] == "load_scene":
                scene_info = msgs.SceneInfo.from_arrow(event["value"])
                scene = Scene(scene_info.name)
                runner.set_difficulty(scene_info.difficulty)
                runner.load_scene(scene)

            elif event["id"] == "joint_commands":
                joint_commands = msgs.JointCommands.from_arrow(event["value"])
                runner.go2.set_target_positions(joint_commands.positions)

            elif event["id"] == "pub_status_tick":
                node.send_output("rtf", pa.array([runner.get_rtf()]))
                node.send_output(
                    "waypoint_mission_complete",
                    pa.array([runner.waypoint_mission.is_complete()]),
                )
                [
                    msgs.Waypoint(
                        status=runner.waypoint_mission.get_waypoint_status(wp),
                        transform=msgs.Transform.from_position_and_quaternion(
                            *wp.get_position()
                        ),
                    ).to_arrow()
                    for wp in runner.waypoint_mission.waypoints
                ]
                node.send_output(
                    "waypoints",
                    msgs.WaypointList(
                        waypoints=[
                            msgs.Waypoint(
                                status=runner.waypoint_mission.get_waypoint_status(wp),
                                transform=msgs.Transform.from_position_and_quaternion(
                                    *wp.get_position()
                                ),
                            )
                            for wp in runner.waypoint_mission.waypoints
                        ]
                    ).to_arrow(),
                )
                robot_pos, robot_quat = runner.get_robot_pose()
                node.send_output(
                    "robot_pose",
                    msgs.Transform.from_position_and_quaternion(
                        robot_pos, robot_quat
                    ).to_arrow(),
                )
                node.send_output(
                    "scene_info",
                    msgs.SceneInfo(
                        name=runner.current_scene.name,
                        difficulty=runner.difficulty,
                    ).to_arrow(),
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
