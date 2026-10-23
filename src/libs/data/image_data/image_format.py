# -*- coding: utf-8 -*-
"""
# ----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 06/02/2024
# version = '0.2'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from typing import Any
from dataclasses import dataclass, field

from src.libs.logger.log import getLogger


@dataclass
class CImageFormat:
    """
    Image format data

    Args:
        fps: Sample rate of the audio.
        width: Image width.
        height: Image height.
        number_channels: Number of audio channels.
    """
    __fps: int
    __width: int
    __height: int
    __number_channels: int

    def __init__(self, fps: int = None, width: int = None,
                 height: int = None, number_channels: int = None):
        self.__fps = fps
        self.__width = width
        self.__height = height
        self.__number_channels = number_channels

        self._logger = getLogger("Image_format")

    @property
    def fps(self) -> float:
        """
        Get the sample rate.

        Returns:
            float: The sample rate of the audio.
        """
        self._logger.trace(f"Accessing fps: {self.__fps}")
        return self.__fps

    @property
    def width(self) -> int:
        """
        Get the width.

        Returns:
            int: width.
        """
        self._logger.trace(f"Accessing width: {self.__width}")
        return self.__width

    @property
    def height(self) -> int:
        """
        Get the height.

        Returns:
            int: height.
        """
        self._logger.trace(f"Accessing height: {self.__height}")
        return self.__height

    @property
    def number_channels(self) -> int:
        """
        Get the number of channels.

        Returns:
            int: The number of audio channels.
        """
        self._logger.trace(f"Accessing number_channels: {self.__number_channels}")
        return self.__number_channels
