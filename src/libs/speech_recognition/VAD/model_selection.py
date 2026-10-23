# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 07/06/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import json
from dataclasses import dataclass

from omegaconf import DictConfig

from .vad_silero import CVADSilero

from src.libs.logger.log import getLogger


@dataclass
class CVADModelSelection:
    """Model selection configuration

    Args:
        cfg_dict: VAD configuration (recognition.vad).
        sample_rate: sample rate of the incoming audio chunks.
    """

    def __init__(self, cfg_dict: dict | DictConfig, sample_rate: int):
        self.models_selection = {}
        config_data = cfg_dict
        vad_silero_recognizer = CVADSilero(silence_thld=config_data["threshold"],
                                           sr=sample_rate,
                                           language=config_data["language"],
                                           window_size_samples=config_data["window_size_samples"],
                                           speech_pad_ms=config_data["speech_pad_ms"],
                                           silence_btwn_words_languages=config_data["Silero"][
                                               "silence_btwn_words_languages"],
                                           min_words_length_languages=config_data["Silero"][
                                               "min_words_length_languages"],
                                           models_folder=config_data["models_folder"])
        self.models_selection["Silero"] = vad_silero_recognizer

        self._logger = getLogger("VAD_Model_selection")

    def get_model(self, model_name: str):
        """
        Get model

        Args:
            model_name: model name

        Raises:
            Exception: Stop the code if the model to select does not exist.
        """
        if model_name in self.models_selection.keys():
            self._logger.debug(f"{model_name} model has been selected")
            return self.models_selection[model_name]
        else:
            self._logger.error(f"Selected model {model_name} is not available")
            raise Exception(f"Throwing error: Selected model {model_name} is not available")
