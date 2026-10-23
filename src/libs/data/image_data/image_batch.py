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
from typing import Any, List

from src.libs.data.data_abstract import AData
from src.libs.data.image_data.image_data import CImageData
from src.libs.logger.log import getLogger


@dataclass
class CImageBatch(AData):
    """
    Image batch data

    Args:
        images: List[CImageData] list of images in the batch
    """
    __images: List[CImageData] = field(default_factory=list, repr=False)

    def __init__(self, images: List[CImageData], id: int):
        super().__init__(id)
        self.__images = images
        self._logger = getLogger(f"Image_batch_{self._id}")

    @property
    def images(self) -> List[CImageData]:
        """
        Get the images in the batch.

        Returns:
            List[CImageData]: The list of images in the batch.
        """
        self._logger.trace(f"Accessing images of batch {self._id}")
        return self.__images

    def get_data(self) -> dict:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        return self.__images
