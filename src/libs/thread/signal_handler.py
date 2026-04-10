# -*- coding: utf-8 -*-
import threading

from src.libs.thread.signal_handler_class import SignalHandler


# Instantiate the global signal handler
_signal_handler_instance = None
_signal_handler_lock = threading.Lock()

def get_signal_handler():
    """
    Get the singleton instance of SignalHandler.

    Returns:
        SignalHandler: The singleton instance of SignalHandler.
    """
    global _signal_handler_instance
    with _signal_handler_lock:
        if _signal_handler_instance is None:
            _signal_handler_instance = SignalHandler()
    return _signal_handler_instance