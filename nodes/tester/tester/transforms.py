import numpy as np
from posetree import Transform
from transforms_py import PyRegistry


class Transforms:
    """Pose tree that retrieves transforms from a PyRegistry."""

    def __init__(self):
        super().__init__()
        self._registry = PyRegistry()

    def get_transform(
        self, parent_frame: str, child_frame: str, timestamp: float = 0
    ) -> Transform:
        """Get a transform from the registry."""
        transform_data = self._registry.get_transform(
            parent_frame, child_frame, timestamp
        )
        if transform_data is None:
            raise KeyError(
                f"No transform found from {parent_frame} to {child_frame} at time {timestamp}"
            )
        tx, ty, tz, qx, qy, qz, qw, _ts, _parent, _child = transform_data
        return Transform.from_position_and_quaternion([tx, ty, tz], [qx, qy, qz, qw])

    def add_transform(
        self,
        transform: Transform,
        timestamp: float,
        parent_frame: str,
        child_frame: str,
    ):
        """Add a transform to the registry."""
        self._registry.add_transform(
            *transform.position, *transform.rotation.as_quat(), timestamp, parent_frame, child_frame
        )

    def distance_to(self, from_frame: str, to_frame: str, timestamp: float = 0) -> float:
        """Compute the Euclidean distance between two frames at a given timestamp."""
        transform = self.get_transform(from_frame, to_frame, timestamp)
        return np.linalg.norm(transform.position)