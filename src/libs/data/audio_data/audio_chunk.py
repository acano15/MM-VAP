# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
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
from dataclasses import dataclass, field

from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger


@dataclass
class CAudioChunk(AData):
    """
    Audio chunk data

    Args:
        buffer: np.ndarray buffer of the chunk
    """
    __buffer: np.ndarray = field(default=None, repr=False)

    def __init__(self, buffer: np.ndarray, id: int):
        super().__init__(id)
        self.__buffer = buffer
        self._logger = getLogger(f"Audio_chunk_{self._id}")

    @property
    def buffer(self) -> np.ndarray:
        """
        Get the chunk buffer.

        Returns:
            np.ndarray: The buffer of the chunk.
        """
        self._logger.trace(f"Accessing buffer of chunk {self._id}")
        return self.__buffer

    def get_data(self) -> np.ndarray:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        return self.__buffer
