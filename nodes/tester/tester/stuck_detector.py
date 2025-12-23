"""Helper class to detect if the robot is stuck."""

import msgs


class StuckDetector:
    """Helper class to detect if the robot is stuck."""

    def __init__(self, threshold: float = 0.1, max_no_progress_time: float = 3.0):
        """Initialize the stuck detector.

        Args:
            threshold: Distance threshold in meters to consider as progress.
            max_no_progress_time: Maximum time in seconds without progress before
                considering the robot as stuck.

        """
        self.threshold = threshold
        self.max_no_progress_time = max_no_progress_time
        self.last_position = None
        self.last_progress_time = None

    def step_is_stuck(self, position: msgs.Transform, current_time: float) -> bool:
        """Update the detector with the current position.

        Returns True if the robot is considered stuck.
        """
        if self.last_position is None or self.last_progress_time is None:
            self.last_position = position
            self.last_progress_time = current_time
            return False

        distance_moved = (
            (position.x - self.last_position.x) ** 2
            + (position.y - self.last_position.y) ** 2
            + (position.z - self.last_position.z) ** 2
        ) ** 0.5

        if distance_moved >= self.threshold:
            self.last_position = position
            self.last_progress_time = current_time
            return False
        else:
            if current_time - self.last_progress_time > self.max_no_progress_time:
                print(
                    "Robot is stuck: no significant movement detected."
                    f" Distance moved: {distance_moved:.3f} m in"
                    f" {current_time - self.last_progress_time:.2f} s."
                )
                return True

        return False
