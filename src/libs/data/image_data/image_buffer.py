# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 23/01/2023
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import numpy as np

from src.libs.data.image_data.image_data import CImageData
from src.libs.data.circular_buffer_abstract import ACircularBuffer
from src.libs.logger.log import getLogger


class CCircularImageBuffer(ACircularBuffer):
    """
    Image buffer module

    Args:
        length (int): Fixed length of the buffer.
        buffer_name (str): Name of the buffer.
        is_streaming (bool): If True, the buffer operates in streaming mode.
        is_replacing (bool): If True, the buffer replace the item.
    """

    def __init__(
            self,
            length: int = 10,
            buffer_name: str = "",
            is_streaming: bool = False,
            is_replacing: bool = True):
        super().__init__(length, buffer_name=buffer_name, is_streaming=is_streaming,
                         is_replacing=is_replacing)

        logger_instance_name = ""
        if self._buffer_name:
            logger_instance_name = f"{self._buffer_name}_"
        self._logger = getLogger(f"{logger_instance_name}Image_buffer")

    def add_item(self, image: CImageData):
        """
        Append one image to the buffer

        Args:
            image: image
        """
        self._logger.debug(f"Storing new image: {image.id}")
        self._store(image)

    def get_item(self) -> CImageData:
        """
        Get frontal image from the buffer

        Returns:
            frontal image.
        """
        return self._retrieve()
