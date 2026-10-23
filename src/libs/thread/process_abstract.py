import os
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import multiprocessing
from abc import ABC, abstractmethod
import time

from src.libs.logger.log import getLogger
from src.libs.thread.signal_handler import get_signal_handler


class AProcess(ABC):
    """
    Abstract base class for processing.

    Args:
        thread_name (str): Name of the process.
    """

    def __init__(self, thread_name: str):
        self._thread_name = thread_name
        self._finish_thread = multiprocessing.Event()
        self._resume_event = multiprocessing.Event()
        self._thread = None
        self._thread_name_lock = multiprocessing.Lock()
        self._is_started = False
        self._condition = multiprocessing.Condition()
        self._updated_thread_name = multiprocessing.Value('b', False)
        self._finish_flag = multiprocessing.Value('b', False)  # Additional flag to track finish state

        self._logger = getLogger("ProcessAbstract")

        signal_handler = get_signal_handler()
        signal_handler.register(self._signal_handler)

    def __del__(self):
        if not self._finish_flag.value:
            self.end_thread()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the logger to avoid pickling errors
        if '_logger' in state:
            del state['_logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._logger = getLogger("ProcessAbstract")

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
        Sets the process name.

        Args:
            name (str): New name for the process.
        """
        self._logger.log_begin("set_thread_name")
        with self._thread_name_lock:
            self._logger.info(f"{self._thread_name}: Process name updated to: {name}")
            self._thread_name = name
            if self._thread:
                self._thread.name = name
            self._updated_thread_name.value = True

        self._logger.log_end("set_thread_name")

    def start_thread(self):
        """
        Starts the process execution.
        """
        self._logger.log_begin("start_thread")
        if not self._is_started:
            self._logger.debug(f"{self._thread_name}: Starting process")
            self._finish_thread.clear()
            self._resume_event.set()
            self._thread = multiprocessing.Process(target=self._run_wrapper, name=self._thread_name)
            self._thread.start()
            self._is_started = True
            self._finish_flag.value = False
            self._logger.debug(f"{self._thread_name} process started")
        else:
            self._logger.warning(f"{self._thread_name}: Process already started")
        self._logger.log_end("start_thread")

    def resume(self):
        """
        Resumes the process execution.
        """
        self._logger.dev(f"{self._thread_name}: Resuming process")
        with self._condition:
            self._resume_event.set()
            self._condition.notify_all()

    def end_thread(self):
        """
        Ends the process execution.
        """
        self._logger.log_begin("end_thread")
        if not self._finish_flag.value:
            self._finish_flag.value = True
            self._logger.debug(f"{self._thread_name}: Ending process")
            self._finish_thread.set()
            self.resume()  # Ensure that the process is not stuck in suspend
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10)
                if self._thread.is_alive():
                    self._logger.error(f"Process {self._thread_name} failed to stop in time")
                    self._thread.terminate()
            self._thread = None
            self._is_started = False
            # Force garbage collection
            gc.collect()
            self._logger.debug(f"{self._thread_name}: Process ended")
        else:
            self._logger.warning(f"{self._thread_name}: Process already ended")
        self._logger.log_end("end_thread")

    def _run_wrapper(self):
        """
        Wrapper for the run method to handle process initialization and finalization.
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
        The method that will be executed by the process.
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
        Initialize the process-specific setup.
        """
        self._logger.log_begin("_init_thread")
        with self._thread_name_lock:
            if self._updated_thread_name.value:
                self._updated_thread_name.value = False

        self._logger.log_end("_init_thread")

    def suspend(self):
        """
        Suspends the process execution.
        """
        self._logger.dev(f"{self._thread_name}: Suspending process")
        with self._condition:
            self._resume_event.clear()
            while not self._resume_event.is_set() and not self._finish_thread.is_set():
                self._condition.wait()

    def suspend_for(self, ms: int):
        """
        Suspends the process execution for a specific amount of time.

        Args:
            ms (int): Time in milliseconds to suspend the process.
        """
        self._logger.dev(f"{self._thread_name}: Suspending process for {ms} ms")
        with self._condition:
            self._resume_event.clear()
            end_time = time.time() + ms / 1000.0
            while time.time() < end_time and not self._resume_event.is_set() and not \
                    self._finish_thread.is_set():
                remaining_time = end_time - time.time()
                self._condition.wait(timeout=remaining_time)
