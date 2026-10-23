# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 13/06/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, List
import threading

from src.libs.logger.log import getLogger
from src.libs.thread.signal_handler import get_signal_handler


class ABatchBuffer(ABC):
    """
    An abstract batch buffer for storing and processing data in batches.

    Args:
        length (int): Fixed length of the buffer.
        batch_size (int): Number of items to batch before processing.
        buffer_name (str): Name of the buffer.
        is_streaming (bool): If True, the buffer operates in streaming mode.
        is_replacing (bool): If True, the buffer replaces the oldest item when full.
    """

    def __init__(
            self,
            length: int,
            batch_size: int,
            buffer_name: str = "",
            is_streaming: bool = False,
            is_replacing: bool = True):
        self._logger = getLogger(self.__class__.__name__)
        self._buffer: Deque[Any] = deque(maxlen=length * batch_size)
        self._mtx_buffer = threading.Lock()
        self._cond_buffer = threading.Condition(self._mtx_buffer)
        self._is_stopping = False
        self._force_incomplete = False
        self._stored_ready = False
        self._batch_size = batch_size
        self._batch_counter = 0
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
        Abstract method to add an item to the buffer.

        Args:
            item (Any): The item to be added to the buffer.

        Returns:
            bool: True if the item was added successfully, False otherwise.
        """
        pass

    def get_batch(self) -> List[Any]:
        """
        Retrieve a batch of items from the buffer.

        Returns:
            List[Any]: The batch of items.
        """
        batch = []
        if not self._interrupted:
            with self._mtx_buffer:
                while len(batch) < self._batch_size and not self._is_stopping and not self._force_incomplete:
                    if len(self._buffer) >= self._batch_size or self._is_stopping or self._force_incomplete:
                        batch.extend([self._buffer.popleft() for _ in range(len(self._buffer))])
                        if len(batch) == self._batch_size:
                            self._logger.debug(f"Retrieved items batch")
                        else:
                            self._logger.debug(f"Retrieved partial batch")

                        if not self._is_streaming:
                            self._cond_buffer.notify_all()
                    else:
                        self._cond_buffer.wait_for(
                            lambda: len(self._buffer) >= self._batch_size or
                                    self._is_stopping or self._force_incomplete)

                # If forced, return the items that are in the buffer
                if self._force_incomplete:
                    if len(self._buffer) > 0 and len(batch) < self._batch_size:
                        while self._buffer and len(batch) < self._batch_size:
                            batch.append(self._buffer.popleft())

                        if len(batch) == self._batch_size:
                            self._logger.debug(f"Retrieved full batch after being forced")
                        else:
                            self._logger.debug(f"Retrieved partial batch due to being forced")

                    if len(self._buffer) > 0:
                        self._logger.warning("Some items were not retrieved and are lost")

                    self._force_incomplete = False
                    self._buffer.clear()

                self._stored_ready = False
        return batch

    def reset(self):
        """
        Reset the buffer.
        """
        self._logger.debug("Resetting buffer")
        self.stop()
        self._batch_counter = 0
        with self._mtx_buffer:
            self._buffer.clear()
        self._is_stopping = False
        self._logger.debug("Buffer reset")

    def stop(self):
        """
        Stop the buffer.
        """
        if not self._is_stopping:
            self._logger.debug("Stopping buffer")
            self._buffer.clear()
            self._is_stopping = True
            with self._cond_buffer:
                self._cond_buffer.notify_all()
            self._logger.debug("Buffer stopped")

    def _store(self, item: Any) -> bool:
        """
        Store an item in the buffer.

        Args:
            item (Any): The item to be stored in the buffer.

        Returns:
            bool: True if the item was stored successfully, False otherwise.
        """
        self._logger.debug(f"Storing new item")
        result = True
        if not self._interrupted:
            with self._mtx_buffer:
                self._is_stopping = False
                if len(self._buffer) == self._buffer.maxlen:
                    if not self._is_streaming:
                        if not self._is_replacing:
                            while len(self._buffer) == self._buffer.maxlen and not self._is_stopping:
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
                    self._logger.debug(f"Stored item")
                    self._batch_counter += 1
                    self._stored_ready = True
                    self._cond_buffer.notify_all()
        else:
            self._logger.warning("Attempt to store an item in a stopped buffer")
            result = False
        return result

    def is_empty(self) -> bool:
        """
        Check if the buffer is empty.

        Returns:
            bool: True if the buffer is empty, False otherwise.
        """
        if len(self._buffer) == 0:
            result = True
        else:
            result = False
        return result

    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        if len(self._buffer) >= self._batch_size:
            result = True
        else:
            result = False
        return result

    def force_incomplete_batch(self):
        """
        Force incomplete batch
        """
        self._logger.debug("Force incomplete batch")
        with self._cond_buffer:
            self._cond_buffer.notify_all()
        self._force_incomplete = True

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
