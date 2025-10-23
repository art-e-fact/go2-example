"""TODO: Add docstring."""

import pyarrow as pa
from dora import Node
import time


def main():
    """TODO: Add docstring."""
    node = Node()

    print("Navigator node started.")

    for event in node:

        if event["type"] == "INPUT":
            if event["id"] == "tick":
                node.send_output("command_2d", pa.array([1,0,0.2]))

            elif event["id"] == "my_input_id":
                # Warning: Make sure to add my_output_id and my_input_id within the dataflow.
                node.send_output(
                    output_id="my_output_id", data=pa.array([1, 2, 3]), metadata={},
                )


if __name__ == "__main__":
    main()
