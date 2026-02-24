# noqa: D100
from pathlib import Path

import msgs
import pyarrow as pa
import pytest
from dora import Node


class TestNode:
    """Dora Node wrapper that allows timeout tracking based on clock messages."""

    def __init__(self):
        """Initialize node state for timeout tracking."""
        self._node = Node()
        self._timeout_from = None
        self._timeout_secs = None

    def reset_timeout(self):
        """Reset the timeout tracking."""
        self._timeout_from = None
        self._timeout_secs = None

    def set_timeout(self, timeout_seconds: float):
        """Set the timeout duration in seconds (relative from the next clock message)."""
        self.reset_timeout()
        self._timeout_secs = timeout_seconds

    def send_output(self, id: str, value: pa.Array, metadata: dict | None = None):
        """Send an output message to the node."""
        self._node.send_output(id, value, metadata)

    def __iter__(self):
        """Iterate over node events.

        `clock` events will be used for tracking timeout.
        """
        for event in self._node:
            # Peek the clock messages to track timeout
            if event["type"] == "INPUT" and event["id"] == "clock":
                now = msgs.Timestamp.from_arrow(event["value"]).float_seconds

                if self._timeout_secs is not None:
                    # Set start time on first clock message
                    # or when the simulation time went backwards (sim reset)
                    if self._timeout_from is None or now < self._timeout_from:
                        self._timeout_from = now
                    elif now - self._timeout_from > self._timeout_secs:
                        raise TimeoutError(
                            f"Node iteration timed out after {self._timeout_secs} seconds."
                        )

            yield event


@pytest.fixture(scope="session")
def session_node():
    """Create a Dora node for the full test session."""
    node = TestNode()
    yield node
    # Send the stop signal to tear down the rest of the nodes
    node.send_output("stop", pa.array([True]))


@pytest.fixture(scope="function")
def node(request, session_node: TestNode):
    """Provide TestNode that will reset the timeout before and after each test.

    Use `@pytest.mark.clock_timeout(seconds)` to set timeout per test.
    """
    # Reset timeout before and after each test
    session_node.reset_timeout()
    # Try to read the clock_timeout marker from the test function
    clock_timeout = request.node.get_closest_marker("clock_timeout")
    if clock_timeout:
        session_node.set_timeout(clock_timeout.args[0])
    yield session_node
    session_node.reset_timeout()


@pytest.fixture(scope="session")
def metrics():
    """Collect test metrics during the test session."""
    metrics = {}

    yield metrics

    workspace_path = Path(__file__).parent.parent.parent.parent
    metrics_path = workspace_path / "metrics.json"
    with open(metrics_path, "w") as f:
        import json

        json.dump(metrics, f, indent=2)
