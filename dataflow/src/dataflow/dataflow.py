import os
import typer
import dora
from dora.builder import DataflowBuilder
from typing_extensions import Annotated
from typing import Optional
from pathlib import Path

workspace_path = Path(__file__).parent.parent.parent.parent
nodes_path = workspace_path / "nodes"
output_path = Path(
    os.getenv("ARTEFACTS_SCENARIO_UPLOAD_DIR", workspace_path / "outputs/artefacts")
)
temp_dataflow_path = output_path / "dataflow.yaml"

policies_folder = Path(__file__).resolve().parent.parent.parent.parent / "policies"
available_policy_folders = (
    sorted([p.name for p in policies_folder.iterdir() if p.is_dir()])
    if policies_folder.exists()
    else []
)


def _exec_dataflow(dataflow: DataflowBuilder, temp_dataflow_path: Path):
    """Build and/or run the dataflow with dora-rs."""
    dataflow.to_yaml(temp_dataflow_path)
    dora.build(str(temp_dataflow_path), uv=True)
    dora.run(str(temp_dataflow_path))


def _create_base_dataflow(policy_path: Path) -> DataflowBuilder:
    dataflow = DataflowBuilder(name="go2-example-dataflow")

    output_path.mkdir(parents=True, exist_ok=True)

    # Set up simulation
    simulation = dataflow.add_node(
        id="simulation",
        path="simulation",
        args="--scene generated_pyramid --use-auto-pilot",
        env={
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "GO2_POLICY_PATH": str(policy_path),
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

    # Set up policy controller
    policy_controller = dataflow.add_node(
        id="policy_controller",
        path="policy_controller",
        env={"GO2_POLICY_PATH": str(policy_path)},
    )
    policy_controller.add_input("observations", "simulation/observations")
    policy_controller.add_input("clock", "simulation/simulation_time")
    policy_controller.add_output("joint_commands")

    return dataflow, simulation, policy_controller


def run_dataflow(
    teleop: Annotated[
        bool, typer.Option(help="Use keyboard teleoperation to control the robot")
    ] = False,
    policy: Annotated[
        Optional[str],
        typer.Option(
            help=(
                "Policy folder name inside 'policies' or absolute path. "
                f"Available: {', '.join(available_policy_folders) if available_policy_folders else 'none detected'}. "
                "Default: GO2_POLICY_PATH env or 'complete'."
            )
        ),
    ] = None,
    test_waypoint_poses: Annotated[
        bool, typer.Option(help="Run the waypoint poses tests")
    ] = False,
    test_waypoint_report: Annotated[
        bool, typer.Option(help="Run the waypoint report tests")
    ] = False,
    test_all: Annotated[bool, typer.Option(help="Run all integration tests")] = False,
):
    """Compose the dataflow, and build/run it with dora-rs."""

    # We either test or teleop for now
    if teleop and (test_waypoint_poses or test_waypoint_report or test_all):
        print("Cannot use teleop and testing options at the same time.")
        return

    # List tests for running
    tests = []
    if test_waypoint_poses or test_all:
        tests.append("test_waypoints_poses.py")
    if test_waypoint_report or test_all:
        tests.append("test_waypoints_report.py")

    if not policy:
        policy = os.getenv("GO2_POLICY_PATH")
        if not policy:
            policy = "complete"

    if policy in available_policy_folders:
        policy = policies_folder / policy

    resolved_policy_path = Path(policy).expanduser().resolve()

    if not resolved_policy_path.exists():
        raise typer.BadParameter(
            f"Policy path '{resolved_policy_path}' does not exist."
        )

    if teleop:
        dataflow, simulation, policy_controller = _create_base_dataflow(
            resolved_policy_path
        )

        teleop_node = dataflow.add_node(
            id="teleop",
            path="teleop",
        )
        teleop_node.add_input("tick", "dora/timer/millis/100")
        teleop_node.add_output("command_2d")
        teleop_node.add_output("load_scene")

        policy_controller.add_input("command_2d", "teleop/command_2d")
        simulation.add_input("load_scene", "teleop/load_scene")

        _exec_dataflow(dataflow, temp_dataflow_path)

    for test in tests:
        dataflow, simulation, policy_controller = _create_base_dataflow(
            resolved_policy_path
        )

        # Add waypoint navigation
        navigator = dataflow.add_node(
            id="navigator",
            path="navigator",
        )
        navigator.add_input("tick", "dora/timer/millis/100")
        navigator.add_input("robot_pose", "simulation/robot_pose")
        navigator.add_input("waypoints", "simulation/waypoints")
        navigator.add_output("command_2d")
        policy_controller.add_input("command_2d", "navigator/command_2d")

        # Add the tester node
        tester = dataflow.add_node(
            id="tester",
            path="pytest",
            args=f"{nodes_path / 'tester/tester' / test} -s --junit-xml={str(output_path / 'tests_junit.xml')}",
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

        _exec_dataflow(dataflow, temp_dataflow_path)


def main():
    typer.run(run_dataflow)


if __name__ == "__main__":
    main()
