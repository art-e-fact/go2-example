# Isaac Sim - Unitree Go2 example

This project demonstrates testing waypoint mission execution with a Unitree Go2 quadruped in different simulated environments.   

<img width="3716" height="2100" alt="go2-example-scenes" src="https://github.com/user-attachments/assets/212c96c9-bbe3-42e5-9a3a-7e28ee00b5f0" />

## Prerequisites
- Isaac Sim 5.0 compatible [hardware](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) and [drivers](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/) (Not mandatory, but the instructions below use `uv`)
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) 

## Setup

```
git clone git@github.com:art-e-fact/go2-example.git
git lfs pull
```

```sh
# Create the virtual environment and install dependencies
uv sync
```

## Run Teleop Demo
Use **WASD** for linear motion and **QE** for turning. Press **R** to reload the scene and **F** to jump to the next one. 
```sh
uv run dataflow  --teleop
```

## Run Tests with dora-rs
This executes all the tests locally.
```sh
# Run test with dora-rs and pytest
uv run dataflow --test-all
```
See `uv run dataflow --help` for all options

## Testing with Artefacts

### Set up the Artefacts Dashboard
Follow these steps to set up your Artefacts project. For more details, refer to the [documentation](https://docs.artefacts.com/getting-started/). 

1. Install the CLI using `pipx` (other installation methods are available).
```sh
sudo apt install pipx
pipx ensurepath
pipx install artefacts-cli
```

2. Create an account at https://app.artefacts.com and log in.
3. Create a new project and follow the authentication instructions provided on the project page.
4. Update [artefacts.yaml](./artefacts.yaml) with your project name.

### Run Tests with Artefacts

```sh
# Launch Isaac Sim and execute multiple waypoint tests
artefacts run waypoint_missions
```
Track the job status on your project page. Test outputs for each scenario will appear there upon completion.




## Project Walkthrough

### Main Tools
 - Isaac Sim for simulation
 - `dora-rs` as the robotics framework
 - PyTorch for executing the control policy

This repository is organized as a Python workspace containing multiple packages.

### Nodes
`dora-rs` nodes are organized as separate Python packages located in `nodes/*`.

 - [`simulation`](./nodes/simulation/) runs the Isaac Sim simulation.
   - Outputs observations and simulation data, such as `robot_pose`, `simulation_time`, and `waypoints`.
   - Listens for low-level joint commands and applies them to the simulated robot.
   - Accepts a `load_scene` input, allowing test nodes to switch scenes without restarting the simulation.
 - [`navigator`](./nodes/navigator/) computes and publishes high-level 2D navigation commands based on the robot's position and waypoints.
 - [`policy_controller`](./nodes/policy_controller/) receives high-level 2D navigation commands from the `navigator` node and outputs low-level joint commands to the `simulation` node.
 - [`tester`](./nodes/tester/) contains test nodes executed via `pytest`.
   - [test_waypoints_poses.py](./nodes/tester/tester/test_waypoints_poses.py) Executes multiple waypoint navigation scenarios, using robot and waypoint position data to verify mission success.
   - [test_waypoints_report.py](./nodes/tester/tester/test_waypoints_report.py) A simplified version of the above test that uses the simulation's internal waypoint mission state to verify success.
 - [`teleop`](./nodes/teleop/) implements keyboard teleoperation control.

### Other Packages
 - [`msgs`](./msgs/) implements necessary messages as Python classes using [`arrow-message`](https://github.com/hennzau/arrow-message).
 - [`dataflow`](./dataflow/) implements a CLI to configure and run `dora-rs` dataflows using the [`dora-rs dataflow builder`](https://github.com/dora-rs/dora/tree/main/examples/python-dataflow-builder). Run `uv run dataflow --help` for options.

### Future Nodes
 - `teleop`: Control the robot via keyboard or gamepad.
 - `dds-transport`: Interface with real Unitree Go2 hardware.


## Training

Policy training is handled in a separate Isaac Lab project: [go2_isaac_lab_env](https://github.com/art-e-fact/go2_isaac_lab_env).

Steps:
 - Follow the instructions in the [go2_isaac_lab_env](https://github.com/art-e-fact/go2_isaac_lab_env) repository to train a new policy.
 - Use `scripts/rsl_rl/play.py` to export the policy.
 - This generates `logs/<checkpoint>/exported/policy.pt` and `logs/<checkpoint>/params/env.yaml`. 
 - Overwrite the files in `./nodes/policy_controller/policy` in this repository with the newly generated files.
 - Test the new policy with `uv run python -m simulation`.



## Development

```sh
# Set up Isaac Sim type hints in VS Code
uv run -m isaacsim --generate-vscode-settings
```

```sh
# Install pre-commit hooks
uv pip install pre-commit
uv run pre-commit install
```
