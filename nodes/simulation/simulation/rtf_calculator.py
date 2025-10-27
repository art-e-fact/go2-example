import time
from collections import deque
from typing import Optional


class RtfCalculator:
    """Calculates the Real-Time Factor (RTF) over a sliding window."""

    def __init__(self, window_size: int = 100, update_interval: int = 100):
        """
        Initializes the RTF calculator.
        Args:
            window_size: The number of simulation steps to include in the sliding window.
            update_interval: The number of steps between RTF calculations.
        """
        self.sim_time_deltas = deque(maxlen=window_size)
        self.wall_time_deltas = deque(maxlen=window_size)
        self.last_wall_time = time.time()
        self.update_interval = update_interval
        self.step_counter = 0
        self.rtf = 0.0

    def step(self, sim_dt: float) -> Optional[float]:
        """
        Records a single simulation step and returns the RTF at a given interval.
        Args:
            sim_dt: The delta time for the current simulation step.
        Returns:
            The calculated RTF if the update interval is reached, otherwise None.
        """
        current_wall_time = time.time()
        wall_time_delta = current_wall_time - self.last_wall_time
        self.last_wall_time = current_wall_time

        self.sim_time_deltas.append(sim_dt)
        self.wall_time_deltas.append(wall_time_delta)

        self.step_counter += 1
        if self.step_counter >= self.update_interval:
            self.step_counter = 0
            total_sim_time = sum(self.sim_time_deltas)
            total_wall_time = sum(self.wall_time_deltas)
            if total_wall_time > 0:
                self.rtf = total_sim_time / total_wall_time
            return self.rtf
        return None

    def reset(self):
        """Resets the calculator's state."""
        self.sim_time_deltas.clear()
        self.wall_time_deltas.clear()
        self.last_wall_time = time.time()
        self.step_counter = 0
        self.rtf = 0.0