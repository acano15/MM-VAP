# -*- coding: utf-8 -*-
import threading
import signal

from src.libs.logger.log import getLogger


class SignalHandler:
    """
    Class for handling signals.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Override the __new__ method to implement the singleton pattern.
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(SignalHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the SignalHandler and sets up the SIGINT signal handler.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._handlers = []
        signal.signal(signal.SIGINT, self._handle_signal)
        self._logger = getLogger(self.__class__.__name__)
        self._initialized = True

    def __del__(self):
        """
        Destructor for SignalHandler.
        Ensures any necessary cleanup is done.
        """
        if len(self._handlers) > 0:
            self._handlers = []

    def register(self, a_handler):
        """
        Registers a new signal handler.

        Args:
            a_handler (callable): The handler function to be called on signal.
        """
        self._handlers.append(a_handler)
        self._logger.debug(f"Added new handler: total {len(self._handlers)}")

    def _handle_signal(self, signum, frame):
        """
        Internal method to handle the signal and call all registered handlers.

        Args:
            signum (int): The signal number.
            frame (frame object): The current stack frame.
        """
        self._logger.debug(f"Executing {len(self._handlers)} signal handlers")
        while self._handlers:
            handler = self._handlers.pop(0)
            try:
                thread = threading.Thread(target=handler, args=(signum, frame))
                thread.start()
            except Exception as e:
                self._logger.error(e)
