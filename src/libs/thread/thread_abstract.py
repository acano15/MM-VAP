import os
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import threading
from abc import ABC, abstractmethod
import time

from src.libs.logger.log import getLogger
from src.libs.thread.signal_handler import get_signal_handler


class AThread(ABC):
    """
    Abstract base class for threading.

    Args:
        thread_name (str): Name of the thread.
    """

    def __init__(self, thread_name: str):
        self._thread_name = thread_name
        self._finish_thread = threading.Event()
        self._resume_event = threading.Event()
        self._thread = None
        self._thread_name_lock = threading.Lock()
        self._is_started = False
        self._condition = threading.Condition()
        self._updated_thread_name = False
        self._finish_flag = False  # Additional flag to track finish state

        self._logger = getLogger("ThreadAbstract")

        signal_handler = get_signal_handler()
        signal_handler.register(self._signal_handler)

    def __del__(self):
        if not self._finish_flag:
            self.end_thread()

    def get_thread_name(self) -> str:
        """
        Gets the thread name.

        Returns:
            str: Name of the thread.
        """
        self._logger.log_begin("get_thread_name")
        with self._thread_name_lock:
            result = self._thread_name

        self._logger.log_end("get_thread_name")
        return result

    def set_thread_name(self, name: str):
        """
        Sets the thread name.

        Args:
            name (str): New name for the thread.
        """
        self._logger.log_begin("set_thread_name")
        with self._thread_name_lock:
            self._logger.info(f"{self._thread_name}: Thread name updated to: {name}")
            self._thread_name = name
            self._thread.name = name
            self._updated_thread_name = True

        self._logger.log_end("set_thread_name")

    def start_thread(self):
        """
        Starts the thread execution.
        """
        self._logger.log_begin("start_thread")
        if not self._is_started:
            self._logger.debug(f"{self._thread_name}: Starting thread")
            self._finish_thread.clear()
            self._resume_event.set()
            self._thread = threading.Thread(target=self._run_wrapper, name=self._thread_name)
            self._thread.start()
            self._is_started = True
            self._finish_flag = False
            self._logger.debug(f"{self._thread_name} thread started")
        else:
            self._logger.warning(f"{self._thread_name}: Thread already started")
        self._logger.log_end("start_thread")

    def resume(self):
        """
        Resumes the thread execution.
        """
        self._logger.dev(f"{self._thread_name}: Resuming thread")
        with self._condition:
            self._resume_event.set()
            self._condition.notify_all()

    def end_thread(self):
        """
        Ends the thread execution.
        """
        self._logger.log_begin("end_thread")
        if not self._finish_flag:
            self._finish_flag = True
            self._logger.debug(f"{self._thread_name}: Ending thread")
            self._finish_thread.set()
            self.resume()  # Ensure that the thread is not stuck in suspend
            if self._thread and self._thread != threading.current_thread() and self._thread.is_alive():
                self._thread.join(timeout=10)
                if self._thread.is_alive():
                    self._logger.error(f"Thread {self._thread_name} failed to stop in time")
            self._thread = None
            self._is_started = False
            # Force garbage collection
            gc.collect()
            self._logger.debug(f"{self._thread_name}: Thread ended")
        else:
            self._logger.warning(f"{self._thread_name}: Thread already ended")
        self._logger.log_end("end_thread")

    def _run_wrapper(self):
        """
        Wrapper for the run method to handle thread initialization and finalization.
        """
        self._logger.log_begin("_run_wrapper")
        try:
            self._init_thread()
            self._run()
        finally:
            self._is_started = False
        self._logger.log_end("_run_wrapper")

    @abstractmethod
    def _run(self):
        """
        The method that will be executed by the thread.
        This method should be overridden by subclasses.
        """
        pass

    @abstractmethod
    def _signal_handler(self):
        """
        The method that will be executed by the signal handler.
        """
        pass

    def _init_thread(self):
        """
        Initialize the thread-specific setup.
        """
        self._logger.log_begin("_init_thread")
        with self._thread_name_lock:
            if self._updated_thread_name:
                self._updated_thread_name = False

        self._logger.log_end("_init_thread")

    def suspend(self):
        """
        Suspends the thread execution.
        """
        self._logger.dev(f"{self._thread_name}: Suspending thread")
        with self._condition:
            while not self._resume_event.is_set() and not self._finish_thread.is_set():
                self._condition.wait()
            self._resume_event.clear()

    def suspend_for(self, ms: int):
        """
        Suspends the thread execution for a specific amount of time.

        Args:
            ms (int): Time in milliseconds to suspend the thread.
        """
        self._logger.dev(f"{self._thread_name}: Suspending thread for {ms} ms")
        with self._condition:
            self._resume_event.clear()
            end_time = time.time() + ms / 1000.0
            while time.time() < end_time and not self._resume_event.is_set() and not \
                    self._finish_thread.is_set():
                remaining_time = end_time - time.time()
                self._condition.wait(timeout=remaining_time)
