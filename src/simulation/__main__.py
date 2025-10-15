import subprocess
from enum import Enum

import typer
from typing_extensions import Annotated

from .scene_config import Scene


class ControlMode(str, Enum):
    """Control mode for the simulation."""

    direct_inference = "direct_inference"
    dds_api = "dds_api"


def check_nvidia_driver():
    required_version = "570"
    try:
        version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        if not version.startswith(required_version):
            yellow = "\033[93m"
            red = "\033[91m"
            reset = "\033[0m"
            print(
                f"{yellow}Warning: NVIDIA driver version is {red}{version}{yellow}, expected {required_version}.x{reset}"
            )
            print(
                "For more info, see: https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html"
            )
    except Exception as e:
        print(f"Could not determine NVIDIA driver version: {e}")


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
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.physx.bundle")

    from .go2_scene import EnvironmentRunner, simulate_unitree_sdk

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
        runner.run()
    elif control_mode == ControlMode.dds_api:
        simulate_unitree_sdk(simulation_app, scene)
    else:
        raise ValueError(f"Unknown control mode: {control_mode}")


if __name__ == "__main__":
    check_nvidia_driver()
    typer.run(simulation)
