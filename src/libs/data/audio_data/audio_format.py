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


import numpy as np
from typing import Any
from dataclasses import dataclass, field

from src.libs.logger.log import getLogger


@dataclass
class CAudioFormat:
    """
    Audio format data

    Args:
        sample_rate: Sample rate of the audio.
        number_channels: Number of audio channels.
        bit_depth: Bit depth of the audio (pyaudio format type).
    """

    def __init__(self, sample_rate: int = None, number_channels: int = None,
                 bit_depth: str = "float"):
        self.__sample_rate = sample_rate
        self.__number_channels = number_channels
        self.__bit_depth = self.__change_format(bit_depth)

        self._logger = getLogger("Audio_format")

    def __change_format(self, config_format: str) -> Any:
        """
        Change audio format.

        Args:
            config_format: Audio format as a string.

        Raises:
            ValueError: If the format is not supported.

        Returns:
            The corresponding pyaudio format.
        """
        format_map = {
            'float': 1,
            'int16': 8,
            'int24': 4,
            'int32': 2,
        }
        format_key = config_format.lower()
        if format_key not in format_map:
            self._logger.error(f"Unsupported format: {config_format}. Supported formats "
                                      f"are: "
                                      f"{', '.join(format_map.keys())}")
            raise ValueError(
                f"Unsupported format: {config_format}. Supported formats are: "
                f"{', '.join(format_map.keys())}")
        return format_map[format_key]

    @property
    def sample_rate(self) -> float:
        """
        Get the sample rate.

        Returns:
            float: The sample rate of the audio.
        """
        self._logger.trace(f"Accessing sample_rate: {self.__sample_rate}")
        return self.__sample_rate

    @property
    def number_channels(self) -> int:
        """
        Get the number of channels.

        Returns:
            int: The number of audio channels.
        """
        self._logger.trace(f"Accessing number_channels: {self.__number_channels}")
        return self.__number_channels

    @property
    def bit_depth(self) -> Any:
        """
        Get the bit depth.

        Returns:
            Any: The bit depth of the audio.
        """
        self._logger.trace(f"Accessing bit_depth: {self.__bit_depth}")
        return self.__bit_depth
