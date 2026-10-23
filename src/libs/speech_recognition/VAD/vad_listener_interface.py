# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 07/06/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class IVADListener(ABC):
    """
    Interface for VAD Listener
    """

    @abstractmethod
    def on_vad(self, index_first_speech: int):
        """
        Called when VAD detects speech.

        Args:
            index_first_speech (int): Index of the first speech detected.
        """
        pass

    @abstractmethod
    def on_sd(self, index_last_speech: int):
        """
        Called when SD detects end of speech.

        Args:
            index_last_speech (int): Index of the last speech detected.
        """
        pass

    @abstractmethod
    def on_processed_chunk(self, vad_results_indexes: List[Tuple[int, bool]], is_last_one: bool):
        """
        Called when a chunk has been processed.

        Args:
            vad_results_indexes (List[Tuple[int, bool]]): List of VAD result indexes with their status.
            is_last_one (bool): Indicates if it is the last chunk.
        """
        pass

    @abstractmethod
    def on_error_vad(self):
        """
        Called when there is an error in VAD.
        """
        pass

    @abstractmethod
    def on_end_vad(self):
        """
        Called when VAD ends.
        """
        pass
