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


from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any
import copy

from src.libs.logger.log import getLogger


@dataclass
class AData(ABC):
    """
    Abstract data class with an optional ID and a private logger.

    Args:
        id: int id of the data.
    """

    def __init__(self, id: int = None):
        self._id = id
        self._logger = getLogger(self.__class__.__name__)

    @property
    def id(self) -> int:
        """
        Get the data id.

        Returns:
            int: The id of the data.
        """
        return self._id

    @abstractmethod
    def get_data(self) -> Any:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        pass

    def copy(self) -> 'AData':
        """
        Create a copy of this instance.

        Returns:
            AData: A new instance of AData with the same properties.
        """
        new_copy = copy.copy(self)

        # Iterate over all fields and deep copy their values, except for the logger
        for f in fields(self):
            if 'logger' not in f.name:
                setattr(new_copy, f.name, copy.deepcopy(getattr(self, f.name)))

        new_copy._logger = self._logger
        return new_copy
