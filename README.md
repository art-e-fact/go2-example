# Isaac Sim - Unitree Go2 example

## Prerequisites
- Isaac Sim 5.0 compatible [hardware](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html) and [driver](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
- [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/) (Not mandatory, but the instructions below are using `uv`)
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) - (Optional) For downloading the photorealistic assets.

## Setup

```sh
# install dependencies (and create virtual environment)
uv sync --dev

# Install this project
uv pip install -e .
```

## Usage
```sh
# To see all options
uv run -m simulation --help

# Control Robot with arrow keys + z/x or gamepad
uv run python -m simulation --scene generated_rails

# Execute waypoint mission automatically
uv run python -m simulation --scene generated_pyramid --use-auto-pilot
```

## Testing
```sh
# Execute tests configured with Artefacts
uv run artefacts run waypoints

# Run test with pytest
uv run pytest
```

## Development
```sh
# Setup isaacsim typehints in VS Code
uv run -m isaacsim --generate-vscode-settings
```