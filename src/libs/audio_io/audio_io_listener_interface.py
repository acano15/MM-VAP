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

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.libs.data.audio_data.audio_chunk import CAudioChunk


class IAudioIOListener(ABC):
    """
    Interface for AudioIO Listener
    """

    @abstractmethod
    def notify_chunk(self, audio_chunk: CAudioChunk, is_last_file_chunk: bool = False):
        """
        Called when AudioIO notify chunk.

        Args:
            audio_chunk (CAudioChunk): Audio chunk.
            is_last_file_chunk: Indicate the last chunk
        """
        pass
