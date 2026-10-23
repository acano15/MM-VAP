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


from dataclasses import dataclass

from src.libs.data.audio_data.audio_chunk import CAudioChunk
from src.libs.data.circular_buffer_abstract import ACircularBuffer
from src.libs.logger.log import getLogger


@dataclass
class CCircularAudioBuffer(ACircularBuffer):
    """Audio circular buffer module

    Args:
        length (int): Fixed length of the buffer.
        buffer_name (str): Name of the buffer.
        is_streaming (bool): If True, the buffer operates in streaming mode.
        is_replacing (bool): If True, the buffer replace the item.
    """

    def __init__(
            self,
            length: int = 1,
            buffer_name: str = "",
            is_streaming: bool = False,
            is_replacing: bool = True):
        super().__init__(length, buffer_name=buffer_name, is_streaming=is_streaming,
                         is_replacing=is_replacing)

        if buffer_name:
            logger_instance_name = f"{buffer_name}_"
        else:
            logger_instance_name = ""
        self._logger = getLogger(f"{logger_instance_name}Audio_buffer")

    def add_item(self, chunk: CAudioChunk):
        """
        Append one chunk to the buffer

        Args:
            chunk: CAudioChunk to append to the buffer
        """
        self._logger.debug(f"Storing new chunk: {chunk.id}")
        self._store(chunk)

    def get_item(self) -> CAudioChunk:
        """
        Get frontal audio chunk

        Returns:
            frontal audio chunk.
        """
        return self._retrieve()
