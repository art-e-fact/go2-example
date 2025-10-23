import weakref
from collections.abc import Callable
from typing import Union

import carb.input
import omni


class InputListener:
    """A minimalistic listener that executes a callback when a key/button is pressed."""

    def __init__(self, key: Union[carb.input.KeyboardInput, carb.input.GamepadInput], callback: Callable):
        """Initializes the listener for a keyboard key or a gamepad button.

        Args:
            key: The keyboard key (carb.input.KeyboardInput) or gamepad button (carb.input.GamepadInput) to listen for.
            callback: The function to call when the key/button is pressed.
        """
        self.key = key
        self.callback = callback

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()

        self._keyboard = None
        self._keyboard_sub = None
        self._gamepad = None
        self._gamepad_sub = None

        if isinstance(key, carb.input.KeyboardInput):
            self._keyboard = self._appwindow.get_keyboard()
            # subscribe to keyboard events using weakref to avoid circular references
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(
                self._keyboard,
                lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
            )
        elif isinstance(key, carb.input.GamepadInput):
            self._gamepad = self._appwindow.get_gamepad(0)
            self._gamepad_sub = self._input.subscribe_to_gamepad_events(
                self._gamepad,
                lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
            )
        else:
            raise TypeError(f"Unsupported key type: {type(key)}. Must be str or carb.input.GamepadInput.")

    def __del__(self):
        """Release the input interface."""
        if self._keyboard_sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if self._gamepad_sub is not None:
            self._input.unsubscribe_from_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None

    def _on_keyboard_event(self, event: carb.input.KeyboardEvent, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == self.key:
                self.callback()
        return True

    def _on_gamepad_event(self, event: carb.input.GamepadEvent, *args, **kwargs):
        """Subscriber callback to when kit is updated."""
        # We only care about button presses, not analog stick movements with values.
        if event.input == self.key and event.value > 0:
            self.callback()
        return True
