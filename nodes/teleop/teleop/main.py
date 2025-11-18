"""Keyboard teleoperation node with holonomic control and scene selection."""

from __future__ import annotations

from typing import Final

from dora import Node
from pynput import keyboard

import msgs

LINEAR_SPEED: Final[float] = 1.2  # m/s
ANGULAR_SPEED: Final[float] = 1.2  # rad/s

MOVEMENT_KEYS = {
    keyboard.KeyCode.from_char("w"),
    keyboard.KeyCode.from_char("a"),
    keyboard.KeyCode.from_char("s"),
    keyboard.KeyCode.from_char("d"),
    keyboard.KeyCode.from_char("q"),
    keyboard.KeyCode.from_char("e"),
}

KEY_NEXT_SCENE = keyboard.KeyCode.from_char("f")
KEY_RELOAD_SCENE = keyboard.KeyCode.from_char("r")

SCENES: Final[list[tuple[str, float]]] = [
    ("generated_pyramid", 0.7),
    ("rail_blocks", 1.0),
    ("stone_stairs", 1.0),
    ("excavator", 1.0),
]


def get_twist(movement_pressed: set[keyboard.KeyCode | keyboard.Key]) -> msgs.Twist2D:
    """Calculate twist based on pressed keys."""
    forward = int(keyboard.KeyCode.from_char("w") in movement_pressed) - int(
        keyboard.KeyCode.from_char("s") in movement_pressed
    )
    strafe = int(keyboard.KeyCode.from_char("a") in movement_pressed) - int(
        keyboard.KeyCode.from_char("d") in movement_pressed
    )
    rotate = int(keyboard.KeyCode.from_char("q") in movement_pressed) - int(
        keyboard.KeyCode.from_char("e") in movement_pressed
    )
    return msgs.Twist2D(
        linear_x=forward * LINEAR_SPEED,
        linear_y=strafe * LINEAR_SPEED,
        angular_z=rotate * ANGULAR_SPEED,
    )


def main() -> None:
    """Entrypoint wiring the teleop publisher."""
    node = Node()

    movement_pressed: set[keyboard.KeyCode | keyboard.Key] = set()
    scene_index = 0

    def publish_scene_info() -> None:
        name, difficulty = SCENES[scene_index]
        scene = msgs.SceneInfo(name=name, difficulty=difficulty)
        node.send_output("load_scene", scene.to_arrow())
        print(f"[teleop] Loading scene '{name}' (difficulty {difficulty})")

    # Initial scene publish
    publish_scene_info()

    try:
        with keyboard.Events() as events:
            while True:
                # Check for dora events with a small timeout to keep the loop spinning
                dora_event = node.next(timeout=0.01)

                if dora_event is not None:
                    if dora_event["type"] == "INPUT":
                        if dora_event["id"] == "tick":
                            twist = get_twist(movement_pressed)
                            node.send_output("command_2d", twist.to_arrow())
                        elif dora_event["id"] == "stop":
                            break

                # Drain all pending keyboard events
                while True:
                    key_event = events.get(0.0)
                    if key_event is None:
                        break

                    if isinstance(key_event, keyboard.Events.Press):
                        key = key_event.key
                        if key in MOVEMENT_KEYS:
                            movement_pressed.add(key)
                        elif key == KEY_NEXT_SCENE:
                            scene_index = (scene_index + 1) % len(SCENES)
                            publish_scene_info()
                        elif key == KEY_RELOAD_SCENE:
                            publish_scene_info()

                    elif isinstance(key_event, keyboard.Events.Release):
                        key = key_event.key
                        if key in MOVEMENT_KEYS:
                            movement_pressed.discard(key)
    except KeyboardInterrupt:
        pass
    finally:
        node.send_output("command_2d", msgs.Twist2D().to_arrow())
