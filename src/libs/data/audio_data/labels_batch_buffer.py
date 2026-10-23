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


from typing import Any
from dataclasses import dataclass

from src.libs.data.data_abstract import AData
from src.libs.data.batch_buffer_abstract import ABatchBuffer
from src.libs.logger.log import getLogger


@dataclass
class CLabelsBatchBuffer(ABatchBuffer):
    """
    Label data batch buffer module

    Args:
        length (int): Fixed length of the buffer.
        batch_size (int): Number of items to batch before processing.
        buffer_name (str): Name of the buffer.
        is_streaming (bool): If True, the buffer operates in streaming mode.
        is_replacing (bool): If True, the buffer replaces the oldest item when full.
    """

    def __init__(self, length: int, batch_size: int, buffer_name: str = "",
                 is_streaming: bool = False, is_replacing: bool = True):
        super().__init__(length, buffer_name=buffer_name, batch_size=batch_size,
                         is_streaming=is_streaming, is_replacing=is_replacing)

        if buffer_name:
            logger_instance_name = f"{buffer_name}_"
        else:
            logger_instance_name = ""
        self._logger = getLogger(f"{logger_instance_name}Labels_batch_buffer")

    def add_item(self, item: AData):
        """
        Append one item to the buffer

        Args:
            item: Item to append to the buffer
        """
        self._logger.debug(f"Storing new label {item.id} to the batch")
        self._store(item)
