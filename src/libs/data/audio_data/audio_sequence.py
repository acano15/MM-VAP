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

from src.libs.data.audio_data.audio_chunk import CAudioChunk
from src.libs.data.data_abstract import AData
from src.libs.logger.log import getLogger


@dataclass
class CAudioSequence(AData):
    """
    Represents a sequence of audio chunks.

    Args:
        audio_chunks (List[CAudioChunk]): List of audio chunks in the sequence.
    """
    __audio_chunks: List[CAudioChunk] = field(init=False, repr=False, default=None)
    __length: int = field(init=False, default=0)

    def __init__(self, audio_chunks: List[CAudioChunk], id: int):
        super().__init__(id)
        self.__audio_chunks = audio_chunks
        self._logger = getLogger(f"Audio_sequence_{self._id}")
        self.__length = len(self.__audio_chunks)

    @property
    def audio_chunks(self) -> List[CAudioChunk]:
        """
        Get the audio chunks.

        Returns:
            List[CAudioChunk]: The list of audio chunks.
        """
        self._logger.trace(f"Accessing audio_chunks: {self.__audio_chunks}")
        return self.__audio_chunks

    @property
    def length(self) -> int:
        """
        Get the length of the audio sequence.

        Returns:
            int: The length of the sequence in terms of the number of audio chunks.
        """
        self._logger.trace(f"Accessing length: {self.__length}")
        return self.__length

    def add_chunk(self, chunk: CAudioChunk) -> None:
        """
        Add an audio chunk to the sequence.

        Args:
            chunk (CAudioChunk): The audio chunk to add.
        """
        self.__audio_chunks.append(chunk)
        self.__length += 1
        self._logger.dev(f"Added chunk with id: {chunk.id}. New length: {self.__length}")

    def remove_chunk(self, chunk: CAudioChunk) -> None:
        """
        Remove an audio chunk from the sequence by chunk ID.

        Args:
            chunk (CAudioChunk): The audio chunk to remove.
        """
        chunk_to_remove = next(
            (item for item in self.__audio_chunks if item.id == chunk.id), None)
        if chunk_to_remove:
            self.__audio_chunks.remove(chunk_to_remove)
            self.__length -= 1
            self.__update_sequence_buffer()
            self._logger.dev(
                f"Removed chunk with id: {chunk.id}. New length: {self.__length}")
        else:
            self._logger.dev(f"Chunk with id: {chunk.id} not found in sequence.")
