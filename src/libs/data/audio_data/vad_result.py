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


from dataclasses import dataclass, field

from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger


@dataclass
class CVADResult(AData):
    """
    Represents the result of Voice Activity Detection (VAD) analysis.

    Args:
        is_speech (bool): Indicates whether there is active voice or not.
        chunk_id (int): ID of the audio chunk.
        confidence (float): Confidence level of the VAD analysis.
    """

    def __init__(self, is_speech: bool, chunk_id: int, confidence: float = None,
                 last_index: bool = False):
        super().__init__(chunk_id)
        self.__is_speech = is_speech
        self.__chunk_id = self._id
        self.__confidence = confidence
        self._is_last_index = last_index
        self._logger = getLogger(f"CVADResult_{self.__chunk_id}")

    @property
    def is_speech(self) -> bool:
        """
        Gets the boolean indicating whether there is active voice or not.

        Returns:
            bool: True if there is active voice, False otherwise.
        """
        self._logger.trace(f"Accessing is_speech: {self.__is_speech}")
        return self.__is_speech

    @property
    def chunk_id(self) -> int:
        """
        Gets the ID of the audio chunk.

        Returns:
            int: ID of the audio chunk.
        """
        self._logger.trace(f"Accessing chunk_id: {self.__chunk_id}")
        return self.__chunk_id

    @property
    def confidence(self) -> float:
        """
        Gets the confidence level of the VAD analysis.

        Returns:
            float: Confidence level of the VAD analysis.
        """
        self._logger.trace(f"Accessing confidence: {self.__confidence}")
        return self.__confidence

    @property
    def is_last_index(self) -> float:
        """
        Gets the boolean indicating whether there is the last index result or not.

        Returns:
            bool: True if there is the last result, False otherwise.
        """
        self._logger.trace(f"Accessing last index: {self._is_last_index}")
        return self._is_last_index

    @is_last_index.setter
    def is_last_index(self, value: bool):
        """
        Set last index

        Args:
            value: value

        """
        self._is_last_index = value

    def get_data(self) -> str:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        return f"{self.__chunk_id}: {int(self.__is_speech)}"
