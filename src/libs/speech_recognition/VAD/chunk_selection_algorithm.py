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


import json
from collections import deque
import math

from src.libs.data.audio_data.vad_result import CVADResult
from omegaconf import DictConfig

from src.libs.logger.log import getLogger


class CVADChunkSelection:
    """
    VAD selection algorithm for VAD results

    Args:
        cfg_dict: modality configuration (recognition.vad or recognition.vvad).
        buffer_length: chunk length in seconds of the modality.
    """

    def __init__(self, cfg_dict: dict | DictConfig, buffer_length: float):
        self.__counter_speech_chunks = 0
        self.__counter_silence_chunks = 0
        self.__is_in_speech_state = False
        self.__results_vad = deque()
        self.__index_first_speech = math.inf
        self.__index_first_silence = math.inf

        config_data = cfg_dict
        self.__init_padding_chunks = config_data["chunk_selection_algorithm"]["init_padding_chunks"]
        self.__end_padding_chunks = config_data["chunk_selection_algorithm"]["end_padding_chunks"]

        use_buffer_size_proportion = config_data["chunk_selection_algorithm"]["use_buffer_size_proportion"]
        if use_buffer_size_proportion:
            self.__speech_buffer_size_in_chunks = self.__get_buffer_size_proportion(buffer_length)
            self.__silence_buffer_size_in_chunks = self.__get_buffer_size_proportion(buffer_length)
        else:
            self.__speech_buffer_size_in_chunks = config_data["chunk_selection_algorithm"][
                "speech_buffer_size_in_chunks"]
            self.__silence_buffer_size_in_chunks = config_data["chunk_selection_algorithm"][
                "silence_buffer_size_in_chunks"]

        self.__percentage_chunks_speech_activation = config_data["chunk_selection_algorithm"][
            "percentage_chunks_speech_activation"]
        self.__percentage_chunks_silence_activation = config_data["chunk_selection_algorithm"][
            "percentage_chunks_silence_activation"]
        self.__speech_threshold = math.ceil(
            self.__speech_buffer_size_in_chunks * self.__percentage_chunks_speech_activation)
        self.__silence_threshold = math.ceil(
            self.__silence_buffer_size_in_chunks * self.__percentage_chunks_silence_activation)

        self.__logger = getLogger("CVADChunkSelection")

    def reset(self):
        """
        Reset
        """
        self._reset_transition_variables()
        self._reset_speech_state()

    def process_vad_results_from_selection(self, result: CVADResult = None) -> dict:
        """
        Process VAD results from selection.

        Args:
            result: VAD result.

        Returns:
            VAD selection result.
        """
        selection_result = {
            "m_chunks_and_speech_status": [],
            "has_vad": None,
            "has_sd": None,
            "m_vad_index": None,
            "m_sd_index": None
        }

        self.__logger.debug(
            "Receiving vad result " + str(result.id) + ". Is speech='" +
            ("true" if result.is_speech else "false") + "' with confidence '" + str(result.confidence) + "'")

        self.__results_vad.append(
            (result.id, result.is_speech))
        if result.is_speech:
            self.__counter_speech_chunks += 1
        else:
            self.__counter_silence_chunks += 1

        if not self.__is_in_speech_state:
            if self.__counter_speech_chunks >= self.__speech_threshold:
                for result_vad in self.__results_vad:
                    if result_vad[1]:
                        self.__index_first_speech = result_vad[0]
                        break

                index_first_speech_padded = max(
                    0,
                    self.__index_first_speech - self.__init_padding_chunks)
                selection_result["has_vad"] = True
                selection_result["m_vad_index"] = index_first_speech_padded
                self.__logger.dev(f"VAD is marked at index {index_first_speech_padded}")

                for result_vad in self.__results_vad:
                    selection_result["m_chunks_and_speech_status"].append(
                        (
                            result_vad[0],
                            result_vad[0] >= selection_result["m_vad_index"]
                        ))

                self.__logger.debug(f"Silence state: More speech results {self.__counter_speech_chunks} "
                                          "than the threshold. Triggering VAD")
                self._reset_transition_variables()
                self.__is_in_speech_state = True
            else:
                self.__logger.dev(f"Silence state: Less speech chunks "
                                        f"{self.__counter_speech_chunks} than the threshold"
                                        f" {self.__speech_threshold}")

            if len(self.__results_vad) >= self.__speech_buffer_size_in_chunks:
                self.__logger.debug("Silence state: Window filled. Removing first element")
                selection_result["m_chunks_and_speech_status"].append(
                    (self.__results_vad[0][0], self.__results_vad[0][1]))
                if self.__results_vad[0][1]:
                    self.__counter_speech_chunks -= 1
                else:
                    self.__counter_silence_chunks -= 1

                self.__results_vad.popleft()
            else:
                self.__logger.dev(f"Silence state: number of results"
                                        f" {len(self.__results_vad)} are less than"
                                        f" {self.__speech_buffer_size_in_chunks}")
        else:
            if self.__counter_silence_chunks >= self.__silence_threshold:
                index_first_silence = self.__results_vad[0][0] - 1 if self.__results_vad[0][0] > 0 else 0
                for it in self.__results_vad:
                    if it[1] == 0:
                        index_first_silence = it[0]
                        break

                last_valid_index = self.__results_vad[-1][0]

                index_first_silence_padded = min(
                    last_valid_index, index_first_silence + self.__end_padding_chunks)
                selection_result["has_sd"] = True
                selection_result["m_sd_index"] = index_first_silence_padded
                self.__logger.dev(f"Silence is marked at index {index_first_silence_padded}")

                for result_vad in self.__results_vad:
                    selection_result["m_chunks_and_speech_status"].append(
                        (
                            result_vad[0],
                            result_vad[0] < selection_result["m_sd_index"]
                        ))

                self.__logger.debug(
                    f"Speech state: More silence results {self.__counter_silence_chunks} than the "
                    "threshold. Triggering silence")
                self._reset_transition_variables()
                self.__is_in_speech_state = False
            else:
                self.__logger.dev(
                    f"Speech state: Less silence chunks "
                    f"{self.__counter_silence_chunks} than the threshold"
                    f" {self.__silence_threshold}")

            if len(self.__results_vad) >= self.__silence_buffer_size_in_chunks:
                self.__logger.debug("Speech state: Window filled. Removing first element")
                selection_result["m_chunks_and_speech_status"].append(
                    (self.__results_vad[0][0], 1))
                if self.__results_vad[0][1]:
                    self.__counter_speech_chunks -= 1
                else:
                    self.__counter_silence_chunks -= 1
                self.__results_vad.popleft()
            else:
                self.__logger.dev(f"Speech state: number of results"
                                        f" {len(self.__results_vad)} are less than"
                                        f" {self.__silence_buffer_size_in_chunks}")

        if result.is_last_index:
            for result_vad in self.__results_vad:
                selection_result["m_chunks_and_speech_status"].append((result_vad[0], result_vad[1]))

            self._reset_transition_variables()
            self._reset_speech_state()
        return selection_result

    def is_in_speech_state(self) -> bool:
        """
        Check if in speech state.

        Returns:
            bool: True if in speech state, False otherwise.
        """
        return self.__is_in_speech_state

    def _reset_transition_variables(self):
        """
        Reset transition variables
        """
        self.__counter_speech_chunks = 0
        self.__counter_silence_chunks = 0
        self.__index_first_speech = math.inf
        self.__index_first_silence = math.inf
        self.__results_vad.clear()

    def _reset_speech_state(self):
        """
        Reset speech state
        """
        self.__is_in_speech_state = False

    @staticmethod
    def __get_buffer_size_proportion(buffer_size: float) -> float:
        """
        Get buffer size proportion.

        Args:
            buffer_size (float): Buffer size in chunks.

        Returns:
            float: Buffer size proportion.
        """
        thresholds = [0.75, 0.5, 0.25, 0.125]
        chunk_counts = [1, 2, 4, 8, 16]

        closest_threshold = min(thresholds, key=lambda x: abs(buffer_size - x))
        closest_index = thresholds.index(closest_threshold)
        return chunk_counts[closest_index]
