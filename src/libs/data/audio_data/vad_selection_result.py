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
from typing import List, Tuple

from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger


@dataclass
class CVADSelectionResult(AData):
    """
    Represents the result of VAD selection.

    Args:
        chunks_and_speech_status: List[Tuple[int, bool]] List of chunk IDs and their speech status.
        has_vad: bool Indicates whether VAD is triggered.
        has_sd: bool Indicates whether silence detection is triggered.
        speech_index: int ID of the audio chunk where speech starts.
        silence_index: int ID of the audio chunk where silence starts.
    """

    def __init__(self, chunks_and_speech_status: List[Tuple[int, bool]], has_vad: bool,
                 has_sd: bool, speech_index: int, silence_index: int, id: int):
        super().__init__(id)
        self.__chunks_and_speech_status = chunks_and_speech_status
        self.__has_vad = has_vad
        self.__has_sd = has_sd
        self.__speech_index = speech_index
        self.__silence_index = silence_index
        self._logger = getLogger(f"VADSelectionResult_{self._id}")

    @property
    def chunks_and_speech_status(self) -> List[Tuple[int, bool]]:
        self._logger.trace(f"Accessing chunks_and_speech_status: {self.__chunks_and_speech_status}")
        return self.__chunks_and_speech_status

    @property
    def has_vad(self) -> bool:
        self._logger.trace(f"Accessing has_vad: {self.__has_vad}")
        return self.__has_vad

    @property
    def has_sd(self) -> bool:
        self._logger.trace(f"Accessing has_sd: {self.__has_sd}")
        return self.__has_sd

    @property
    def speech_index(self) -> int:
        self._logger.trace(f"Accessing speech_index: {self.__speech_index}")
        return self.__speech_index

    @property
    def silence_index(self) -> int:
        self._logger.trace(f"Accessing silence_index: {self.__silence_index}")
        return self.__silence_index

    def get_data(self) -> dict:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        result = {"speech": {self.__speech_index: self.__has_vad},
                  "silence": {self.__silence_index: self.__has_sd}}
        return result
