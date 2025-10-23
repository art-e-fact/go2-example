# TODO: This won't handle obstacles higher than the robot base frame.
import omni.physx
from isaacsim.util.debug_draw import _debug_draw


import numpy as np


class HeightScanGrid:
    def __init__(self, debug_vis: bool = False, skip_path_with: str = "/Go2/"):
        # TODO: Read these from env.yaml
        self.resolution = 0.06
        self.size = [0.8, 0.6]
        self.clip = (-0.5, 0.05)
        # (Experimental) The amount if rays we cast per cell. This can help with catching thin obstacles like rails.
        self.sample_count = 6
        # How much higher the rays start above the robot (should be lower the obstacles above, like cealing, but high enough for hitting steep stairs from above)
        self.source_offset = 1.0
        self._skip_path_with = skip_path_with
        self._raw_height_data = np.zeros(
            (
                int(self.size[0] / self.resolution) + 1,
                int(self.size[1] / self.resolution) + 1,
            )
        )
        # Default setting in isaaclab/envs/mdp/observations.py height_scan()
        self._offset = 0.5
        self._x_steps, self._y_steps = self._raw_height_data.shape
        self._x_step_size = self.size[0] / (self._x_steps - 1)
        self._y_step_size = self.size[1] / (self._y_steps - 1)
        self._debug_vis = debug_vis
        if debug_vis:
            self._debug_sources = np.zeros((self._x_steps, self._y_steps, 3))

    def scan(self, position: np.ndarray, yaw: float = 0.0):
        # Simulate a height scan by filling the height_data with random values
        if self._debug_vis:
            self._draw_height_data()

        self._raw_height_data.fill(1.0 + self.source_offset)  # Reset height data with 1.0
        for iy in range(self._y_steps):
            for ix in range(self._x_steps):
                offset_x = -(self.size[0] / 2) + (ix * self._x_step_size)
                offset_y = -(self.size[1] / 2) + (iy * self._y_step_size)
                # Rotate the offsets based on the yaw
                offset_x_rotated = offset_x * np.cos(yaw) - offset_y * np.sin(yaw)
                offset_y_rotated = offset_x * np.sin(yaw) + offset_y * np.cos(yaw)
                # Calculate the source position
                source = position + np.array([offset_x_rotated, offset_y_rotated, self.source_offset])
                if self._debug_vis:
                    self._debug_sources[ix, iy] = source

                def callback(hit):
                    return self._report_all_hits(hit, ix, iy)

                for _ in range(self.sample_count):
                    offset_x = np.random.uniform(-self._x_step_size / 2, self._x_step_size / 2)
                    offset_y = np.random.uniform(-self._y_step_size / 2, self._y_step_size / 2)
                    omni.physx.get_physx_scene_query_interface().raycast_all(
                        source + np.array([offset_x, offset_y, 0.0]),
                        (0.0, 0.0, -1.0),
                        100.0,
                        callback,
                )

    def _draw_height_data(self):
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_points()
        draw.clear_lines()

        data = self.get_height_data() + self._offset + self.source_offset

        sources = np.transpose(self._debug_sources, (1, 0, 2)).reshape(-1, 3)
        hits = sources.copy()
        hits[:, 2] -= data
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.draw_points(
            hits,
            [(1.0, 0.0, 0.0, 1.0)] * len(hits),
            [15] * len(hits),
        )
        draw.draw_lines(
            sources,
            hits,
            [(0.0, 1.0, 0.0, 1.0)] * len(hits),
            [1] * len(hits),
        )

    def _report_all_hits(self, hit_info, i, j) -> bool:
        # Ignore the robot
        if self._skip_path_with not in hit_info.rigid_body:
            # Hits are not reported in order, so we need to check if the hit is closer than the current height
            distance = hit_info.distance
            if self._raw_height_data[i, j] > distance:
                self._raw_height_data[i, j] = distance
            return True

    def get_height_data(self) -> np.ndarray:
        # Transpose the height data because the Isaac Lab RayScan uses yx indexing by default
        data = self._raw_height_data.T
        # Flatten the height data to a 1D array
        return np.clip(data.flatten() - self._offset - self.source_offset, *self.clip)