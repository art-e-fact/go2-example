from pathlib import Path
import numpy as np
import yaml


class PolicyConfig:
    def __init__(self, config_path: Path):
        """
        Initializes the Config with a configuration file.

        Args:
            config_path (Path): Path to the .yaml file.
        """
        # Load the model and configuration
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
            def ignore_unknown(self, node) -> None:
                return None

            def tuple_constructor(loader, node) -> tuple:
                # The node is expected to be a sequence node
                return tuple(loader.construct_sequence(node))

        SafeLoaderIgnoreUnknown.add_constructor(
            "tag:yaml.org,2002:python/tuple", SafeLoaderIgnoreUnknown.tuple_constructor
        )
        SafeLoaderIgnoreUnknown.add_constructor(
            None, SafeLoaderIgnoreUnknown.ignore_unknown
        )
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

    @property
    def dt(self) -> float:
        return self.config.get("sim").get("dt")

    @property
    def decimation(self) -> int:
        return self.config.get("decimation")
    
    @property
    def action_scale(self) -> float:
        return 0.2  # TODO: Wind out where to read this from

    # TODO: Read these from env.yaml
    # Default joint positions from the policy's env.yaml (converted from list to np.array)
    default_joint_pos = np.array(
        [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
    )
