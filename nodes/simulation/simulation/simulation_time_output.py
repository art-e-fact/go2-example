import msgs
from dora import Node
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import isaacsim.core.api


class SimulationTimeOutput:
    """Registers a physics step callback and sends simulation time."""

    def __init__(self, node: Node, world: "isaacsim.core.api.World"):
        self.node = node
        self.world = world
        self.simulation_time = 0.0
        self.previous_time_step_index = -1
        world.add_physics_callback("simulation_time_output", self.on_physics_step)

    def on_physics_step(self, dt: float):
        # Reset the couter if the simulation has been reset
        if self.world.current_time_step_index < self.previous_time_step_index:
            self.simulation_time = 0.0
        self.previous_time_step_index = self.world.current_time_step_index
        self.simulation_time += dt
        timestamp = msgs.Timestamp.from_float_seconds(self.simulation_time)
        self.node.send_output("simulation_time", timestamp.to_arrow())