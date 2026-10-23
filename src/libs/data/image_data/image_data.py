# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/05/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from dataclasses import dataclass, field
import cv2
import numpy as np

from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger


@dataclass
class CImageData(AData):
    """
    Image data

    Args:
        image: np.ndarray frame
    """
    __image: np.ndarray = field(default=None, repr=False)

    def __init__(self, frame: np.ndarray, id: int):
        super().__init__(id)
        self.__frame = frame
        self.__is_converted = True

        self._logger = getLogger(f"Image_{self._id}")

    @property
    def frame(self) -> np.ndarray:
        """
        Get the frame.

        Returns:
            np.ndarray: The frame.
        """
        self._logger.trace(f"Accessing frame with id: {self._id}")
        return self.__frame

    def get_data(self) -> dict:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        return self.__frame

    def copy_image(self):
        """
        Creates a copy of the image data (frame) to allow it to be saved independently.

        This method deep copies the numpy array representing the image frame and returns
        a new CImageData object with the copied image, ensuring no reference is maintained to the original.

        Returns:
            CImageData: A new instance with a copied frame.
        """
        copied_frame = self.__frame.copy()
        return CImageData(frame=copied_frame, id=self._id)

    def __getstate__(self):
        """
        Control what gets pickled.
        Exclude the logger from the serialized state.
        """
        state = self.__dict__.copy()
        if '_logger' in state:
            del state['_logger']
        return state

    def __setstate__(self, state):
        """
        Restore the state and reinitialize the logger after unpickling.
        """
        self.__dict__.update(state)
        self._logger = getLogger(f"Image_{self._id}")
