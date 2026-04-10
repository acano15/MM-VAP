# -*- coding: utf-8 -*-
import threading
from collections import deque
from typing import Any, Deque, Optional

from src.libs.logger.log import getLogger
from src.libs.thread.signal_handler import get_signal_handler


class CCircularBuffer:
    """
    A thread-safe circular buffer that holds a fixed number of elements.
    When the buffer is full, adding a new element will overwrite the oldest one.

    Attributes:
        length (int): Maximum number of items in the buffer.
        is_streaming (bool): If True, producer never blocks when full.
        is_replacing (bool): If True, replaces oldest item when full.
    """

    def __init__(self, length: int, name: str = "", is_streaming: bool = False, is_replacing: bool = False) -> None:
        self.logger = getLogger(self.__class__.__name__)

        self.length = length
        self.is_replacing = is_replacing
        self.is_streaming = is_streaming
        
        self.counter = 0
        self.buffer = deque(maxlen=length)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.is_stopped = False
        self.is_interrupted = False
        self.item_ready = False

        signal_handler = get_signal_handler()
        signal_handler.register(self._signal_handler)

    def stop(self):
        """Stops buffer activity and unblocks waiting threads."""
        if not self.is_stopped:
            self.logger.debug("Stopping circular buffer")
            self.counter = 0
            self.is_stopped = True
            with self.condition:
                self.condition.notify_all()
            self.buffer.clear()
            self.logger.debug("Circular buffer stopped")

    def reset(self):
        """Resets buffer to initial state."""
        self.logger.info("Resetting circular buffer")
        self.stop()
        with self.lock:
            self.buffer.clear()
            self.is_stopped = False
            self.is_interrupted = False
        self.logger.info("Circular buffer reset complete")

    def add(self, item: Any, name: Optional[str] = None) -> bool:
        """Adds an item to the buffer safely.

        Args:
            item (Any): Data to add to the buffer.
            name (Optional[str]): Optional name or tag associated with the item.

        Returns:
            bool: True if item was stored, False otherwise.
        """
        result = False
        if self.is_interrupted:
            self.logger.warning("Cannot add item: buffer interrupted")
            return result

        with self.lock:
            # Handle full buffer conditions
            if len(self.buffer) == self.length:
                if not self.is_streaming:
                    if not self.is_replacing:
                        self.logger.debug("Buffer full — waiting for consumer")
                        while len(self.buffer) == self.length and not self.is_stopped:
                            self.condition.wait()
                    else:
                        removed = self.buffer.popleft()
                        self.logger.warning(f"Replacing oldest (non-streaming): {removed}")
                else:
                    if self.is_replacing:
                        removed = self.buffer.popleft()
                        self.logger.warning(f"Replacing oldest (streaming): {removed}")
                    else:
                        self.logger.warning("Streaming mode full — discarding new item")
                        result = False

            if name is not None:
                item_id = name
            else:
                item_id = self.counter

            self.buffer.append({"id": item_id, "data": item})
            self.logger.debug(f"Item added to buffer id: {item_id}")
            self.counter += 1

            self.item_ready = True
            self.condition.notify_all()
            result = True

        return result

    def get(self) -> Optional[Any]:
        """Retrieves the oldest item from the buffer."""
        if self.is_interrupted:
            self.logger.warning("Attempt to retrieve from interrupted buffer")
            return None

        item = None
        while item is None and not self.is_stopped:
            with self.lock:
                if len(self.buffer) > 0:
                    wrapped_item = self.buffer.popleft()
                    item_id = wrapped_item["id"]
                    item = wrapped_item["data"]
                    self.logger.debug(f"Item retrieved from buffer id: {item_id}")
                    if not self.is_streaming:
                        self.condition.notify_all()
                else:
                    self.condition.wait_for(lambda: self.item_ready or self.is_stopped)
                    self.item_ready = False
        return item

    def clear(self) -> None:
        """
        Clears all items from the buffer.
        """
        with self.lock:
            self.buffer.clear()
            self.logger.debug("Buffer cleared.")

    def is_full(self) -> bool:
        """Returns True if buffer is full."""
        return len(self.buffer) >= self.length

    def is_empty(self) -> bool:
        """
        Checks if the buffer is empty.

        Returns:
            bool: True if the buffer is empty, False otherwise.
        """
        with self.lock:
            return len(self.buffer) == 0

    def __len__(self) -> int:
        """
        Returns the current number of items in the buffer.

        Returns:
            int: Number of items in the buffer.
        """
        return len(self.buffer)

    def remaining_items_in_buffer(self) -> int:
        """
        Returns the number of items remaining to be processed in the buffer.

        Returns:
            int: Number of remaining items.
        """
        with self.lock:
            return len(self.buffer)

    def unlock(self) -> None:
        """
        Unlocks the buffer in case it is stuck.
        """
        with self.condition:
            self.item_ready = True
            self.condition.notify_all()
        self.logger.debug("Buffer unlocked manually")

    def _signal_handler(self, signum, frame):
        """Called automatically when user presses Ctrl+C or process is killed."""

        self.logger.warning("The signal handler has been invoked")
        self.is_interrupted = True
        self.stop()
