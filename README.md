# Isaac Sim - Unitree Go2 example

This project demos testing waypoint mission execution by a Unitree Go2 quadruped in different simulated environments.   

## Prerequisites
- Isaac Sim 5.0 compatible [hardware](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) and [driver](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/) (Not mandatory, but the instructions below are using `uv`)
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) 

## Setup

```
git clone git@github.com:art-e-fact/go2-example.git
git lfs pull
```

```sh
# create the virtual environment and install dependencies
uv sync
```

## Run teleop demo
Use **WASD** for linear motion and **QE** for turning. **R** reloads the scene and **F** jumps to the next one. 
```sh
uv run dataflow  --teleop
```


## Testing with Artefacts

Follow the instructions at [docs.artefacts.com](https://docs.artefacts.com/getting-started/) to set-up the project. 

```sh
# Launch Isaac Sim and execute multiple waypoint tests
uvx --from artefacts-cli artefacts run waypoint_missions
```

## Run tests with dora-rs
This will execute all the tests without parameterization in `artefacts.yaml`
```sh
# Run test with dora-rs and pytest
uv run dataflow --test-all
```
See `uv run dataflow --help` for all options


## Project walkthrough

### Main tools:
 - Isaac Sim for simulation
 - `dora-rs` as the robotics framework
 - PyTorch executing the control policy

This repo is organized as a Python workspace with multiple Python packages.

### Nodes
The `dora-rs` nodes are organized as separate Python packages under `nodes/*`

 - [`simulation`](./nodes/simulation/) runs the Isaac Sim simulation
   - It outputs observations and information about the simulation like `robot_pose`, `simulation_time`, `waypoints`.
   - It listens to the low-level joint commands and applies them on the simulated robot.
   - Also accepts `load_scene` input that allows the test nodes to load different scenes without restarting the simulation.
 - [`navigator`](./nodes/navigator/) using the robot position and the waypoint positions, it computes and publishes the high-level 2D navigation commands.
 - [`policy_controller`](./nodes/policy_controller/) takes the high-level 2D navigation commands from the `navigator` nodes and outputs the low-level joint commands to the `simulation` node
 - [`tester`](./nodes/tester/) contains the test nodes that should be executed with `pytest`
   - [test_waypoints_poses.py](./nodes/tester/tester/test_waypoints_poses.py) Executes multiple waypoint navigation scenarios and uses the robot and waypoint position data to determine if the waypoint mission was successful.
   - [test_waypoints_report.py](./nodes/tester/tester/test_waypoints_report.py) The simplified version of the test above, that uses the internal waypoint mission state from the simulation to determine if the waypoint mission was successful.
 - [`teleop`](./nodes/teleop/) implements keyboard teleop control

### Other packages
 - [`msgs`](./msgs/) Implements the necessary messages as python classes using [`arrow-message`](https://github.com/hennzau/arrow-message) 
 - [`dataflow`](./dataflow/) Using the [`dora-rs dataflow builder`](https://github.com/dora-rs/dora/tree/main/examples/python-dataflow-builder) implements a CLI to configure and run `dora-rs` dataflows. Run `uv run dataflow --help` to see all options.

### Future nodes
 - `teleop` for controlling the robot with keyboard or gamepad
 - `dds-transport` for interfacing the real Unitree Go2 hardware


## Training

Policy training is separated in a standard Isaac Lab project: https://github.com/art-e-fact/go2_isaac_lab_env.

Steps:
 - Follow the instructions in [go2_isaac_lab_env](https://github.com/art-e-fact/go2_isaac_lab_env) train the new policy
 - Use `scripts/rsl_rl/play.py` to export the trained policy.
 - This will generate `logs/<checkpoint>/exported/policy.pt` and `logs/<checkpoint>/params/env.yaml`. 
 - Override these files in the `./nodes/policy_controller/policy` of this repo.
 - Try the new policy with `uv run python -m simulation`



## Development

```sh
# Setup isaacsim type hints in VS Code
uv run -m isaacsim --generate-vscode-settings
```

```sh
# Install pre-commit hooks
uv pip install pre-commit
uv run pre-commit install
```