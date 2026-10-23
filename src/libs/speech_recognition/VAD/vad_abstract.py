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


import numpy as np
from threading import Thread
from abc import ABC, abstractmethod

from src.libs.processing.audio_processing.audio_processing import CAudioProcessing

from src.libs.logger.log import getLogger


class AVoiceActivityDetector(ABC):
    """
    Voice Activity Detector abstract module

    Args:
        frames: original audio samples
        silences: silences detected with its confidence
    """

    def __init__(self, models_folder: str = None):
        self._path_models = models_folder or str(
            Path(__file__).resolve().parents[4] / "assets" / "models" / "speech" / "VAD")
        self._frames = None
        self._threshold = None
        self._audio_processing = CAudioProcessing()

        self._logger = getLogger("VVAD_detector")

    @abstractmethod
    def inference(self, frames):
        pass

    def start(self, frames: np.ndarray):
        """
        Recognition in threading mode

        Args:
            frames: numpy audio samples
        """
        Thread(target=self.inferce, args=(frames,)).start()
        return self

    def _from_silences_to_labels(self) -> bool:
        """
        Get binary labels from silence
        """
        if len(self.silences) > 0:
            result = 1
        else:
            result = 0
        return result

    @staticmethod
    def _get_silences_from_detection(speech_timestamps) -> dict:
        """
        Get silences from detection

        Args:
            speech_timestamps: detection after overpassing the confidence

        Returns:
            Silence dictionary containing timestamp labels.
        """
        silences_dict = {}
        for i, speech_timestamp in enumerate(speech_timestamps):
            silences_dict[i] = (speech_timestamp['start'], speech_timestamp['end'])
        return silences_dict
