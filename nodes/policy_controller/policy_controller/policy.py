"""Policy module for robot control.

This module provides the Policy class for loading and executing a trained
policy model to generate robot actions based on observations and commands.
"""

from pathlib import Path
import msgs
import numpy as np
import torch

from .policy_config import PolicyConfig


class Policy:
    """A policy for robot control using a trained neural network model.

    This class loads a trained policy model and configuration, then generates
    robot actions based on observations and commands.

    Attributes
    ----------
    model : torch.jit.ScriptModule
        The loaded TorchScript model.
    config : PolicyConfig
        The policy configuration loaded from a YAML file.

    Methods
    -------
    reset() -> None
        Reset the policy's internal state.
    forward(observation: msgs.Observations, command: msgs.Twist2D) -> torch.Tensor
        Generate robot actions based on observations and commands.

    """

    def __init__(self, model_path: Path, config_path: Path):
        """Initialize the Policy with a model and configuration.

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
        """Reset the policy's internal state."""
        self._previous_action = np.zeros(12)

    def forward(
        self, observation: msgs.Observations, command: msgs.Twist2D
    ) -> torch.Tensor:
        """Get the action from the policy based on the observation.

        Args:
            observation (dict): The observation input for the policy.
            command (msgs.Twist2D): The current command input for the policy.

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
        obs[9:12] = np.array([command.linear_x, command.linear_y, command.angular_z])
        # Joint states
        obs[12:24] = observation.joint_positions - self.config.default_joint_pos
        obs[24:36] = observation.joint_velocities
        obs[36:48] = self._previous_action
        obs[48:202] = observation.height_scan

        with torch.no_grad():
            observation = torch.from_numpy(obs).view(1, -1).float()
            action = self.model(observation).detach().view(-1).numpy()
        self._previous_action = action

        target_positions = self.config.default_joint_pos + (
            action * self.config.action_scale
        )

        return target_positions
