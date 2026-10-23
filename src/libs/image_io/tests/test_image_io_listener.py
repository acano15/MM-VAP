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


from src.libs.image_io.image_io_listener_interface import IImageIOListener
from src.libs.data.image_data.image_data import CImageData
from src.libs.logger.log import getLogger, set_logger_level


class CImageIOListenerTest(IImageIOListener):
    """
    ImageIO Listener
    """

    def __init__(self, log_level: str = "INFO"):
        self._logger = getLogger("ImageIO_Listener_test")
        set_logger_level("ImageIO_Listener_test", log_level)
        self._logger.info("ImageIO Listener test instance created")

    def notify_frame(self, image_frame: CImageData, is_last_file_frame: bool = False):
        """
        Called when imageIO detects speech.

        Args:
            image_frame (CImageData): image frame.
            is_last_file_chunk: Indicate the last frame
        """
        if image_frame:
            self._logger.info(f"Received image frame {image_frame.id}")
