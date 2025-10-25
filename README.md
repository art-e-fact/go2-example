# Isaac Sim - Unitree Go2 example

## Prerequisites
- Isaac Sim 5.0 compatible [hardware](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) and [driver](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/) (Not mandatory, but the instructions below are using `uv`)
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) - (Optional) For downloading the photorealistic assets.

## Setup

```sh
# create the virtual environment
uv venv --seed --python 3.11

# Install dora-rs
uv pip install dora-rs-cli

# Install all nodes
uv run dora build dataflow.test.yaml --uv
```


## Testing with Artefacts

Follow the instructions at [docs.artefacts.com](https://docs.artefacts.com/getting-started/) to set-up the project. 

```sh
# Launch Isaac Sim and execute multiple waypoint tests
uv run artefacts run waypoint_missions
```

## Run tests with dora-rs
This will execute all the tests without parameterization in `artefacts.yaml`
```sh
# Run test with dora-rs and pytest
uv run dora run dataflow.test.yaml --uv
```

## Development
```sh
# Setup isaacsim typehints in VS Code
uv run -m isaacsim --generate-vscode-settings
```

## Training

Policy training is separated in a standard Isaac Lab project: https://github.com/art-e-fact/go2_isaac_lab_env.

Steps:
 - Follow the instructions in [go2_isaac_lab_env](https://github.com/art-e-fact/go2_isaac_lab_env) train the new policy
 - Use `scripts/rsl_rl/play.py` to export the trained policy.
 - This will generate `logs/<checkpoint>/exported/policy.pt` and `logs/<checkpoint>/params/env.yaml`. 
 - Override these files in the `./policy` of this repo.
 - Try the new policy with `uv run python -m simulation` 