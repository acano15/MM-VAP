# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 06/02/2023
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from typing import Deque, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
import signal
import threading

from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger
from src.libs.thread.signal_handler import get_signal_handler


@dataclass
class ACircularBuffer(ABC):
    """
    An abstract circular buffer for storing data
    """

    _counter = 0

    def __init__(self, length: int, buffer_name: str = "", is_streaming: bool = False,
                 is_replacing: bool = True):
        self._logger = getLogger(self.__class__.__name__)

        self._buffer: Deque[Optional[Any]] = deque(maxlen=length)
        self._mtx_buffer = threading.Lock()
        self._cond_buffer = threading.Condition(self._mtx_buffer)
        self._is_stopping = False
        self._stored_ready = False
        self._buffer_length = length
        self._buffer_name = buffer_name
        self._is_streaming = is_streaming
        self._is_replacing = is_replacing

        self._interrupted = False
        signal_handler = get_signal_handler()
        signal_handler.register(self._signal_handler)

    def __del__(self):
        self.stop()

    @abstractmethod
    def add_item(self, item: Any) -> bool:
        """
        Abstract method to add an item to the buffer
        """
        pass

    @abstractmethod
    def get_item(self):
        """
        Abstract method to get an item from the buffer
        """
        pass

    def reset(self):
        """
        Reset the buffer
        """
        self._logger.debug("Resetting buffer")
        self.stop()
        self._counter = 0
        with self._mtx_buffer:
            self._buffer.clear()
            self._buffer = deque(maxlen=self._buffer_length)
            self._is_stopping = False
        self._logger.debug("Buffer reset")

    def stop(self):
        """
        Stop the buffer
        """
        if not self._is_stopping:
            self._logger.debug("Stopping buffer")
            self._buffer.clear()
            self._is_stopping = True
            with self._cond_buffer:
                self._cond_buffer.notify_all()
            self._logger.debug("Buffer stopped")

    def _store(self, item: AData) -> bool:
        """
        Store an item in the buffer.

        Args:
            item (AData): The item to be stored in the buffer.

        Returns:
            bool: True if the item was stored successfully, False otherwise.
        """
        self._logger.debug(f"Storing new item: {item.id}")
        result = True
        if not self._interrupted:
            with self._mtx_buffer:
                self._is_stopping = False
                if len(self._buffer) == self._buffer.maxlen:
                    if not self._is_streaming:
                        if not self._is_replacing:
                            while len(
                                    self._buffer) == self._buffer.maxlen and not self._is_stopping:
                                self._logger.debug(
                                    "Non-streaming mode: buffer is full. Waiting until an item is consumed")
                                self._cond_buffer.wait()
                        else:
                            front_item = self._buffer.popleft()
                            self._logger.warning(
                                f"Non-streaming mode: buffer full. Discarding oldest item"
                                f" {front_item.id}")
                    else:
                        if self._is_replacing:
                            front_item = self._buffer.popleft()
                            self._logger.warning(
                                f"Streaming mode: buffer full. Discarding oldest item"
                                f" {front_item.id}")
                        else:
                            result = False
                            self._logger.warning(
                                f"Streaming mode: buffer full. Discarding item {item.id}")

                if result:
                    self._buffer.append(item)
                    self._logger.debug(f"Stored item: {item.id}")
                    self._counter += 1
                    self._stored_ready = True
                    self._cond_buffer.notify_all()
        else:
            self._logger.warning("Attempt to store an item in a stopped buffer")
            result = False
        return result

    def _retrieve(self) -> Optional[Any]:
        """
        Retrieve an item from the buffer
        """
        front_item = None
        if not self._interrupted:
            while front_item is None and not self._is_stopping:
                with self._mtx_buffer:
                    if not self.is_empty():
                        front_item = self._buffer.popleft()
                        self._logger.debug(f"Retrieved item: {front_item.id}")
                        if not self._is_streaming:
                            self._cond_buffer.notify_all()
                    else:
                        self._logger.debug("Warning: Attempt to retrieve from empty buffer")
                        self._cond_buffer.wait_for(
                            lambda: self._stored_ready or self._is_stopping)
                        self._stored_ready = False

        if not front_item:
            self._logger.warning("Attempt to retrieve from a stopped buffer")
        return front_item

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty
        """
        if len(self._buffer) == 0:
            result = True
        else:
            result = False
        return result

    def is_full(self) -> bool:
        """
        Check if the buffer is empty
        """
        if len(self._buffer) >= self._buffer_length:
            result = True
        else:
            result = False
        return result

    def _signal_handler(self, signum, frame):
        """
        Signal handler

        Args:
            signum:
            frame:
        """
        self._logger.warning("The signal handler has been invoked")
        self._interrupted = True
        self.stop()
