# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/05/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from src.libs.data.batch_buffer_abstract import ABatchBuffer
from src.libs.data.audio_data.audio_chunk import CAudioChunk
from src.libs.logger.log import getLogger


class CAudioBatchBuffer(ABatchBuffer):
    """
    Audio batch buffer for storing and processing data in batches.

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
        self._logger = getLogger(f"{logger_instance_name}Audio_batch_buffer")

    def add_item(self, audio: CAudioChunk):
        """
        Append one audio to the buffer

        Args:
            audio: audio
        """
        self._logger.debug(f"Storing new audio {audio.id} to the batch")
        self._store(audio)
