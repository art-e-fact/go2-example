"""Module to process node input events and print received messages."""

from dora import Node
import pytest


# @pytest_asyncio.fixture
# async def node():
#     """Create a Dora node for testing."""
#     print(">>>>>>>>>> Creating Dora node for testing")
#     return Node()

@pytest.fixture
def node():
    """Create a Dora node for testing."""
    print(">>>>>>>>>> Creating Dora node for testing")
    yield Node()

# @pytest.mark.asyncio
# # @pytest.mark.timeout(1)
# async def test_hears_input_event(node: Node):
#     print("Starting test_hears_input_event")
#     while True:
#         print("Waiting to receive event...")
#         event = await node.recv_async()
#         print(f">>>>>>>>>>>>>> Received event: {event}")
#         if event is None:
#             continue
#         if event["type"] == "INPUT":
#             if event["id"] == "speecha":
#                 message = event["value"][0].as_py()
#                 print(f"""I heard {message} from {event["id"]}""")
#                 assert message is not None
#                 return
#         await asyncio.sleep(0.1)

@pytest.mark.timeout(5)
def test_hears_input_event_sync(node: Node):
    print("Starting test_hears_input_event")
    for event in node:
        print(f">>>>>>>>>>>>>> Received event: {event}")
        if event is None:
            continue
        if event["type"] == "INPUT":
            if event["id"] == "speech":
                message = event["value"][0].as_py()
                print(f"""I heard {message} from {event["id"]}""")
                assert message is not None
                return

# def main():
#     """Listen for input events and print received messages."""
#     node = Node()
#     for event in node:
#         if event["type"] == "INPUT":
#             message = event["value"][0].as_py()
#             print(f"""I heard {message} from {event["id"]}""")

# if __name__ == "__main__":
#     main()
