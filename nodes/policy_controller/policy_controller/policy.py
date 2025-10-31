from pathlib import Path
import msgs
import numpy as np
import torch

from .policy_config import PolicyConfig


class Policy:
    def __init__(self, model_path: Path, config_path: Path):
        """
        Initializes the Policy with a model and configuration.

        Args:
            model_path (Path): Path to the .pt file.
            config_path (Path): Path to the .yaml file.
        """
        # Load the model and configuration
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = torch.jit.load(model_path)
        self.config = PolicyConfig(config_path)
        self._previous_action = np.zeros(12)
    
    def reset(self):
        self._previous_action = np.zeros(12)

    def forward(self, observation: msgs.Observations, command: msgs.Twist2D) -> torch.Tensor:
        """
        Gets the action from the policy based on the observation.

        Args:
            observation (dict): The observation input for the policy.

        Returns:
            torch.Tensor: The action output from the policy.
        """
        obs = np.zeros(202)
        # Base lin vel
        obs[:3] = observation.lin_vel
        # Base ang vel
        obs[3:6] = observation.ang_vel
        # Gravity
        obs[6:9] = observation.gravity
        # Command
        obs[9:12] = np.array(
            [command.linear_x, command.linear_y, command.angular_z]
        )
        # Joint states
        obs[12:24] = observation.joint_positions - self.config.default_joint_pos
        obs[24:36] = observation.joint_velocities
        obs[36:48] = self._previous_action
        obs[48:202] = observation.height_scan

        with torch.no_grad():
            observation = torch.from_numpy(obs).view(1, -1).float()
            action = self.model(observation).detach().view(-1).numpy()
        self._previous_action = action

        target_positions = self.config.default_joint_pos + (action * self.config.action_scale)

        return target_positions

