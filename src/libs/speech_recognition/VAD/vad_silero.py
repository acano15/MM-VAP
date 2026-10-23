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


import json
from collections import OrderedDict
import platform
import torch

from .vad_abstract import AVoiceActivityDetector
from .utils_vad import (init_jit_model,
                       get_speech_timestamps,
                       get_number_ts,
                       get_language,
                       get_language_and_group,
                       save_audio,
                       read_audio,
                       VADIterator,
                       collect_chunks,
                       drop_chunks,
                       Validator,
                       OnnxWrapper)


class CVADSilero(AVoiceActivityDetector):
    """
    VAD module for Silero

    Args:
        silence_thld: threshold to accept the inference
        sr: original sample rate of the frames
        language: language selected
        window_size_samples: Audio chunks of window_size_samples size are fed to the silero VAD model
        speech_pad_ms: final speech chunks are padded by speech_pad_ms each side
        silence_btwn_words_languages: minimum silence (ms) between words per language
        min_words_length_languages: minimum word length (ms) per language
        models_folder: folder containing silero_vad.jit
    """
    silences = []

    def __init__(self, silence_thld: float = 0.7, sr: int = 16000, language: str = "EN",
                 window_size_samples: int = 512, speech_pad_ms: int = 30,
                 silence_btwn_words_languages: dict = None,
                 min_words_length_languages: dict = None, models_folder: str = None):
        super().__init__(models_folder)
        self.silence_btwn_words_languages = OrderedDict(
            silence_btwn_words_languages or {"ES": 100, "EN": 100})
        self.min_words_length_languages = OrderedDict(
            min_words_length_languages or {"ES": 50, "EN": 50})
        self._logger.set_new_name("Silero_VVAD_detector")
        self._threshold = silence_thld
        self._sr = sr
        self._window_size_samples = window_size_samples
        self._speech_pad_ms = speech_pad_ms
        self._min_speech_duration_ms = self.min_words_length_languages[language]
        self._min_silence_duration_ms = self.silence_btwn_words_languages[language]

        # TODO: GPU version not working in Windows
        self._device = 'cuda' if (torch.cuda.is_available() and platform.system() != "Windows") \
            else 'cpu'

        self._vad_model = init_jit_model(os.path.join(self._path_models, 'silero_vad.jit'),
                                           device=self._device)
        # self.vad_iterator = VADIterator(self._vad_model, threshold=silence_thld,
        # sampling_rate=self._sr,
        # min_silence_duration_ms=self._min_silence_duration_ms,
        # speech_pad_ms=speech_pad_ms)
        self._logger.info("Model Silero VAD loaded successfully")

    def inference(self, frames: np.ndarray, return_seconds: bool = False,
                  return_label: bool = False) -> tuple:
        """
        Inference the VAD from audio samples

        Args:
            frames: numpy audio samples
            return_seconds: if true return seconds
            return_label: if true return label

        Returns:
            tuple with the result and the speech probabilities.
        """
        self._logger.log_begin("inference")
        if not torch.is_tensor(frames):
            try:
                frames = torch.Tensor(frames)
            except BaseException:
                self._logger.error("Audio cannot be casted to tensor. Cast it manually")
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")
        else:
            frames = frames.copy()

        if torch.cuda.is_available() and self._device == "cuda":
            frames = frames.cuda()

        self._logger.dev("Silero inference")
        speech_prob = 0.0
        try:
            speech_prob = self._vad_model(frames, self._sr).item()
        except Exception as e:
            speech_prob = -1
            self._logger.error(f"Error when computing silero prediction: {e}")

        if speech_prob >= self._threshold:
            result = 1
        else:
            result = 0
        self._logger.log_end("inference")
        return result, speech_prob

    def inference_silence(self, frames: np.ndarray, return_seconds: bool = False,
                          return_label: bool = False) -> bool:
        """
        Inference silence

        Args:
            frames: numpy audio samples
            return_seconds: if true return seconds
            return_label: if true return label
        """
        silences = self.__get_silences_marks(frames, self._sr, return_seconds)
        if return_label:
            result = self.__from_silences_to_labels()
        else:
            result = silences
        return result

    def reset_state(self):
        """
        Reset state
        """
        self._vad_model.reset_states()

    def __get_silences_marks(self, frames: np.ndarray, return_seconds: bool) -> dict:
        """
        Recognize all the voice activity over a threshold

        Args:
            frames: numpy audio samples
            sr: sample rate
            return_seconds: if true return seconds

        Returns:
            Silence dictionary containing timestamp labels.
        """
        frames_copy = frames.copy()
        frames_copy = self._audio_processing.int2float(frames_copy)
        frames_copy_data = torch.from_numpy(frames_copy)
        if (torch.cuda.is_available()):
            frames_copy_data = frames_copy_data.cuda()
        speech_timestamps = get_speech_timestamps(frames_copy_data, self._vad_model,
                                                  sampling_rate=self._sr,
                                                  min_speech_duration_ms=self._min_speech_duration_ms,
                                                  min_silence_duration_ms=self._min_silence_duration_ms,
                                                  window_size_samples=self._window_size_samples,
                                                  speech_pad_ms=self._speech_pad_ms,
                                                  return_seconds=return_seconds,
                                                  visualize_probs=False)
        silences = self._get_silences_from_detection(speech_timestamps)
        return silences
