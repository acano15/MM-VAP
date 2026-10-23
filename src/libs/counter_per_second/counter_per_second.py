# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/01/2022
# version = '0.1'
# ---------------------------------------------------------------------------
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import time
from threading import Thread, Event

from src.libs.logger.log import getLogger
from src.libs.thread.thread_abstract import AThread


class CCountsPerSec(AThread):
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """
    _date_format = "%d/%m/%Y %H:%M:%S"

    def __init__(self, name: str = "CountsPerSec"):
        self._logger = getLogger(name)
        super().__init__(f"{name}")
        self._time_init = time.time()
        self._counter = 0
        self._num_occurrences = 0
        self._running = Event()
        self._running.set()

    def start(self):
        """
        Start counting in a background thread.
        """
        self.start_thread()

    def stop(self):
        """
        Stop the counter
        """
        self._running.clear()
        self.resume()
        self.end_thread()

    def _run(self):
        """
        The method that will be executed by the thread
        """
        self._init_thread()
        while not self._finish_thread.is_set():
            self.suspend_for(1000)
            elapsed_time = time.time() - self._time_init

            if elapsed_time > 0:
                self._counter = round(self._num_occurrences / elapsed_time, 2)
                self._logger.info(f"Counts per second: {self._counter}")
                self._time_init = time.time()
                self._num_occurrences = 0

    def increment(self, count_unity: int = 1):
        """
        Increment the count of occurrences.

        Args:
            count_unity (int): The number of occurrences to increment by.
        """
        self._num_occurrences += count_unity

    def get_counter(self) -> float:
        """
        Get the current counts per second (FPS).

        Returns:
            float: The counts per second.
        """
        return self._counter

    def get_date_format(self) -> str:
        """
        Get date_format

        Returns:
            string date format.
        """
        return self._date_format

    def _signal_handler(self, signum, frame):
        """
        Signal handler

        Args:
            signum:
            frame:
        """
        self._logger.warning("The signal handler has been invoked")
        self.stop()
