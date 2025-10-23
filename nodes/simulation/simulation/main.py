"""TODO: Add docstring."""

import pyarrow as pa
from dora import Node


from enum import Enum

import typer
from typing_extensions import Annotated

from simulation.scene_config import Scene
from simulation.check_nvidia_driver import check_nvidia_driver


class ControlMode(str, Enum):
    """Control mode for the simulation."""

    direct_inference = "direct_inference"
    dds_api = "dds_api"


def simulation(
    scene: Annotated[
        Scene,
        typer.Option(prompt=True, help="Environment to load", case_sensitive=False),
    ],
    control_mode: Annotated[
        str,
        typer.Option(
            help="`direct_inference` runs the policy in the same process while `dds_api` waits for standard Untree SDK commands"
        ),
    ] = ControlMode.direct_inference.value,
    rerun: Annotated[
        bool, typer.Option(help="Log visualization data to Rerun")
    ] = False,
    use_video_stream: Annotated[
        bool,
        typer.Option(help="Use video stream instead of images for logging to Rerun"),
    ] = False,
    use_auto_pilot: Annotated[
        bool, typer.Option(help="Use the auto pilot to navigate waypoints")
    ] = False,
    rrd_path: Annotated[
        str,
        typer.Option(
            help="Stream the Rerun logs into this file. (Instead of streaming on gRPC)."
        ),
    ] = None,
    difficulty: Annotated[
        float,
        typer.Option(help="Difficulty level for generated scenes (0.0 to 1.0)"),
    ] = 0.5,
):
    node = Node()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.physx.bundle")

    from simulation.go2_scene import EnvironmentRunner, simulate_unitree_sdk

    if control_mode == ControlMode.direct_inference:
        runner = EnvironmentRunner(
            simulation_app,
            scene,
            rerun,
            use_video_stream,
            use_auto_pilot,
            rrd_path,
            difficulty,
        )
        runner.initialize()

        while runner.simulation_app.is_running():
            runner.step()
            node.send_output("rtf", pa.array([runner.get_rtf()]))
            node.send_output(
                "waypoint_mission_complete",
                pa.array([runner.waypoint_mission.is_complete()]),
            )

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
                    scene_name = event["value"][0].as_py()
                    print(f"Node received load_scene command: {scene_name}")
                    scene = Scene(scene_name)
                    runner.load_scene(scene)

                elif event["id"] == "command_2d":
                    command_2d = event["value"].to_pylist()
                    print(f"Node received command_2d: {command_2d}")
                    x, y, yaw = command_2d
                    runner.set_command(x, y, yaw)

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

    # elif control_mode == ControlMode.dds_api:
    #     simulate_unitree_sdk(simulation_app, scene)
    # else:
    #     raise ValueError(f"Unknown control mode: {control_mode}")


def main():
    check_nvidia_driver()
    typer.run(simulation)


if __name__ == "__main__":
    main()
