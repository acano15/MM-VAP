import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import threading
import logging

try:
    from pynput import keyboard
except (ImportError, OSError):
    keyboard = None

from src.libs.logger.log import getLogger


class CKeyboardHandler:
    """
    Class for handling keyboard events.
    """

    def __init__(self):
        try:
            self._logger = getLogger("KeyboardHandler")
        except RuntimeError:
            self._logger = logging.getLogger("KeyboardHandler")
        self._listeners = {}
        self._handlers = {}
        self._stop_event = threading.Event()
        self._listener = None
        if keyboard is not None:
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        self._pressed_keys = set()

    def __del__(self):
        self.stop()

    def stop(self):
        """
        Stops the keyboard listener.
        """
        if self._listener:
            self._listener.stop()
        self._stop_event.set()

    def register(self, key, handler):
        """
        Registers a new key handler.

        Args:
            key (keyboard.Key or keyboard.KeyCode): The key to handle.
            handler (callable): The handler function to be called on key press.
        """
        if keyboard is None:
            return
        self._listeners[key] = handler
        self._logger.debug(f"Registered key {key} with handler {handler}")

    def _on_press(self, key):
        """
        Internal method to handle key press events.

        Args:
            key (keyboard.Key or keyboard.KeyCode): The key that was pressed.
        """
        self._logger.debug(f"User pressed {key}")

    def _on_release(self, key):
        """
        Internal method to handle key release events.

        Args:
            key (keyboard.Key or keyboard.KeyCode): The key that was released.
        """
        self._logger.debug(f"User released {key}")
        if key in self._listeners.keys():
            if isinstance(key, keyboard.Key):
                self._logger.info(f"Key {key} detected")
            elif isinstance(key, keyboard.KeyCode) and hasattr(key, 'char'):
                self._logger.info(f"Key char {key.char} detected")

            self._listeners[key]()

keyboard_handler = CKeyboardHandler()
