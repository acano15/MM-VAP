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

from src.libs.data.image_data.image_data import CImageData


class IImageIOListener(ABC):
    """
    Interface for ImageIO Listener
    """

    @abstractmethod
    def notify_frame(self, frame: CImageData, is_last_file_frame: bool = False):
        """
        Called when ImageIO notify frame.

        Args:
            frame (CImageData): Image frame.
            is_last_file_chunk: Indicate the last frame
        """
        pass
