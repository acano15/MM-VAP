# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 11/06/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from typing import List, Tuple

from src.libs.audio_io.audio_io_listener_interface import IAudioIOListener
from src.libs.data.audio_data.audio_chunk import CAudioChunk
from src.libs.logger.log import getLogger, set_logger_level


class CAudioIOListenerTest(IAudioIOListener):
    """
    AudioIO Listener
    """

    def __init__(self, log_level: str = "INFO"):
        self._logger = getLogger("AudioIO_Listener_test")
        set_logger_level("AudioIO_Listener_test", log_level)

        self._logger.info("AudioIO Listener test instance created")

    def notify_chunk(self, audio_chunk: CAudioChunk, is_last_file_chunk: bool = False):
        """
        Called when AudioIO detects speech.

        Args:
            audio_chunk (CAudioChunk): Audio chunk.
            is_last_file_chunk: Indicate the last chunk
        """
        self._logger.info(f"Received audio chunk {audio_chunk.id}")
