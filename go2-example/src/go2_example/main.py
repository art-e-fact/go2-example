from copy import deepcopy

import typer
import dora
from dora.builder import DataflowBuilder
from typing_extensions import Annotated
from pathlib import Path


def exec_dataflow(
    dataflow: DataflowBuilder, build: bool, run: bool, temp_dataflow_path: Path
):
    """Build and/or run the dataflow with dora-rs."""
    dataflow.to_yaml(temp_dataflow_path)
    if build:
        dora.build(str(temp_dataflow_path), uv=True)
    if run:
        dora.run(str(temp_dataflow_path))


def run_dataflow(
    test_waypoint_poses: Annotated[
        bool, typer.Option(help="Run the waypoint poses tests")
    ] = False,
    test_waypoint_report: Annotated[
        bool, typer.Option(help="Run the waypoint report tests")
    ] = False,
    test_all: Annotated[bool, typer.Option(help="Run all integration tests")] = False,
    build: Annotated[
        bool, typer.Option(help="Build the dataflow with `dora build` before running")
    ] = True,
    run: Annotated[
        bool,
        typer.Option(help="Run the dataflow with `dora run` (turn off to build only)"),
    ] = True,
):
    """Compose the dataflow, and build/run it with dora-rs."""

    # List tests for running
    tests = []
    if test_waypoint_poses or test_all:
        tests.append("test_waypoints_poses.py")
    if test_waypoint_report or test_all:
        tests.append("test_waypoints_report.py")
    
    if not tests:
        print("No tests selected to run. Use --help for options.")
        # TODO: run default dataflow without tester node (and optionally with teleop)

    for test in tests:
        dataflow = DataflowBuilder(name="go2-example-dataflow")
        workspace_path = Path(__file__).parent.parent.parent.parent
        output_path = workspace_path / "outputs/artefacts"
        nodes_path = workspace_path / "nodes"
        temp_dataflow_path = output_path / "dataflow.yaml"

        output_path.mkdir(parents=True, exist_ok=True)

        # Set up simulation
        simulation = dataflow.add_node(
            id="simulation",
            path="simulation",
            build=f"pip install -e {nodes_path / 'simulation'}",
            args="--scene generated_pyramid --use-auto-pilot",
            env={
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "OUTPUT_DIR": str(output_path),
            },
        )
        simulation.add_input("pub_status_tick", "dora/timer/millis/200")
        simulation.add_input("joint_commands", "policy_controller/joint_commands")
        simulation.add_output("robot_pose")
        simulation.add_output("waypoints")
        simulation.add_output("scene_info")
        simulation.add_output("rtf")
        simulation.add_output("observations")
        simulation.add_output("simulation_time")

        policy_controller = dataflow.add_node(
            id="policy_controller",
            path="policy_controller",
            build=f"pip install -e {nodes_path / 'policy_controller'}",
        )
        policy_controller.add_input("observations", "simulation/observations")
        policy_controller.add_input("clock", "simulation/simulation_time")
        policy_controller.add_input("command_2d", "navigator/command_2d")
        policy_controller.add_output("joint_commands")

        # Add waypoint navigation
        navigator = dataflow.add_node(
            id="navigator",
            path="navigator",
            build=f"pip install -e {nodes_path / 'navigator'}",
        )
        navigator.add_input("tick", "dora/timer/millis/100")
        navigator.add_input("robot_pose", "simulation/robot_pose")
        navigator.add_input("waypoints", "simulation/waypoints")
        navigator.add_output("command_2d")

        # Add the tester node
        tester = dataflow.add_node(
            id="tester",
            path="pytest",
            args=f"{nodes_path / 'tester/tester' / test} -s --junit-xml={str(output_path / 'tests_junit.xml')}",
            build=f"pip install -e {nodes_path / 'tester'}",
        )
        tester.add_input("waypoints", "simulation/waypoints")
        tester.add_input("scene_info", "simulation/scene_info")
        tester.add_input("robot_pose", "simulation/robot_pose")
        tester.add_input("clock", "simulation/simulation_time")
        tester.add_output("stop")
        tester.add_output("load_scene")

        # Tear down nodes when the tester node is done
        simulation.add_input("stop", "tester/stop")
        policy_controller.add_input("stop", "tester/stop")
        navigator.add_input("stop", "tester/stop")

        # Allow the tester to load scenes in the simulation
        simulation.add_input("load_scene", "tester/load_scene")

        exec_dataflow(dataflow, build, run, temp_dataflow_path)


def main():
    typer.run(run_dataflow)


if __name__ == "__main__":
    main()
