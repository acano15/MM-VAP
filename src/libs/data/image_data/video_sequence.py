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
from typing import List

from src.libs.data.image_data.image_batch import CImageBatch
from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger


@dataclass
class CVideoSequence(AData):
    """
    Represents a sequence of image batches.

    Args:
        image_batches (List[CImageBatch]): List of image batches in the sequence.
    """
    __image_batches: List[CImageBatch] = field(init=False, repr=False, default=None)
    __length: int = field(init=False, default=0)

    def __init__(self, image_batches: List[CImageBatch], id: int):
        super().__init__(id)
        self.__image_batches = image_batches
        self.__length = len(self.__image_batches)
        self._logger = getLogger(f"Video_sequence_{self._id}")

    @property
    def image_batches(self) -> List[CImageBatch]:
        """
        Get the image batches.

        Returns:
            List[CImageBatch]: The list of image batches.
        """
        self._logger.trace(f"Accessing image_batches: {self.__image_batches}")
        return self.__image_batches

    @property
    def length(self) -> int:
        """
        Get the length of the video sequence.

        Returns:
            int: The length of the sequence in terms of the number of image batches.
        """
        self._logger.trace(f"Accessing length: {self.__length}")
        return self.__length

    def add_batch(self, batch: CImageBatch) -> None:
        """
        Add an image batch to the sequence.

        Args:
            batch (CImageBatch): The image batch to add.
        """
        self.__image_batches.append(batch)
        self.__length += 1
        self._logger.dev(f"Added batch with id: {batch.id}. New length: {self.__length}")

    def remove_batch_by_id(self, batch_id: int) -> None:
        """
        Remove an image batch from the sequence by batch ID.

        Args:
            batch_id (int): The ID of the image batch to remove.
        """
        batch_to_remove = next((batch for batch in self.__image_batches if batch.id == batch_id), None)
        if batch_to_remove:
            self.__image_batches.remove(batch_to_remove)
            self.__length -= 1
            self._logger.dev(f"Removed batch with id: {batch_id}. New length: {self.__length}")
        else:
            self._logger.dev(f"Batch with id: {batch_id} not found in sequence.")
