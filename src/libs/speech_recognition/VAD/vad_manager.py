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
from typing import Any

from .model_selection import CVADModelSelection
from .chunk_selection_algorithm import CVADChunkSelection
from src.libs.data.audio_data.audio_buffer import CCircularAudioBuffer
from src.libs.data.audio_data.labels_buffer import CCircularLabelsBuffer
from src.libs.data.audio_data.audio_chunk import CAudioChunk
from src.libs.data.audio_data.vad_result import CVADResult
from src.libs.data.audio_data.vad_selection_result import CVADSelectionResult
from src.libs.thread.thread_abstract import AThread
from src.libs.logger.log import getLogger
from omegaconf import DictConfig

# DEFINITIONS AND MACROS
BUFFER_LENGTH = 50


class CVADManager(AThread):
    """
    VAD Manager WT

    Args:
        cfg_dict: root Hydra configuration (uses recognition, preprocessing and io).
        mode: [AudioChunks, Labels] define the mode, AudioChunks compute the VAD labels while
        Labels get from outside
        type: modality type [vad or vvad]
    """

    def __init__(self, cfg_dict: DictConfig, mode: str = "Labels", type: str = "vad"):
        self._mode = mode
        thread_name = "CVADManager"
        self._buffer = None
        self._listeners = []
        self._is_stopped = True
        self._is_active = True
        super().__init__(f"Thread_{thread_name}")

        vad_config = cfg_dict.recognition[type]
        processing_name = "audio_processing" if type.lower() == "vad" else "image_processing"
        buffer_length = cfg_dict.preprocessing[processing_name].buffer_length
        self._vad_selection_manager = CVADChunkSelection(vad_config, buffer_length)
        if self._mode == "AudioChunks":
            self._vad_recognizer = CVADModelSelection(
                vad_config, cfg_dict.io.audio_io.audio_format.sample_rate).get_model(
                vad_config.type)
            self._buffer = CCircularAudioBuffer(
                BUFFER_LENGTH,
                buffer_name=thread_name,
                is_streaming=False,
                is_replacing=False)
        elif self._mode == "Labels":
            self._buffer = CCircularLabelsBuffer(
                BUFFER_LENGTH,
                buffer_name=thread_name,
                is_streaming=False,
                is_replacing=False)
        self._last_id = -1

        self._logger = getLogger(thread_name)

    def __del__(self):
        self.stop()
        super().__del__()

    def start(self):
        """
        Starts the execution of the manager.
        """
        self._logger.log_begin("start")
        if self._is_stopped:
            self._logger.info(f"Starting {self._thread_name}")
            self.start_thread()
            self._last_id = -1
            self._is_stopped = False
            self._is_active = True
            self._logger.info(f"{self._thread_name} started")
        else:
            self._logger.warning(f"Trying to start a thread {self._thread_name} that has "
                                       f"already started")
        self._logger.log_end("start")

    def stop(self):
        """
        Stop

        """
        self._logger.log_begin("stop")
        if not self._is_stopped:
            self._logger.info(f"Stopping {self._thread_name}")
            self._is_active = False
            self._is_stopped = True
            self._listeners.clear()
            self.resume()
            if self._buffer:
                self._buffer.stop()
            self.end_thread()
            self._logger.info(f"{self._thread_name} stopped")
        self._logger.log_end("stop")

    def reset(self):
        """
        Reset
        """
        self._logger.log_begin("reset")
        self.stop()
        self._vad_selection_manager.reset()
        self._buffer.reset()
        self.start()
        self._logger.log_end("reset")

    def add_listener(self, listener: Any) -> bool:
        """
        Adds a listener.

        Args:
            listener (IVADListener): Listener to be added.

        Returns:
            bool: True if listener is added successfully.
        """
        self._logger.debug("Adding new listener to VAD manager")
        self._listeners.append(listener)
        self._logger.debug(
            f"New listener added to VAD manager. Total number of listeners: "
            f"{len(self._listeners)}")
        return True

    def notify_chunk(self, chunk: CAudioChunk, is_last_file_chunk: bool) -> bool:
        """
        Notifies a new chunk.

        Args:
            chunk (CAudioChunk): Received chunk.
            is_last_file_chunk: Indicate the last chunk

        Returns:
            bool: True if chunk is stored successfully, False otherwise.
        """
        self._logger.debug(f"VAD notify: storing chunk {chunk.id}")
        result = False
        if is_last_file_chunk:
            self.set_last_index(chunk.id)

        if self._buffer:
            self._buffer.add_item(chunk)
            self.resume()
            result = True
        else:
            self._logger.warning("VAD manager buffer is null")
        self._logger.debug(f"Chunk stored with result: {result}")
        return result

    def notify_vad_result(self, result: Any, is_last_file_result: bool) -> bool:
        """
        Notifies a new VAD result.

        Args:
            result (CVADResult): Received result.
            is_last_file_result: Indicate the last result

        Returns:
            bool: True if result is stored successfully, False otherwise.
        """
        self._logger.debug(f"VAD notify: storing vad result {result.id}")
        is_stored = False
        if is_last_file_result:
            self.set_last_index(result.id)

        if self._buffer:
            self._buffer.add_item(result)
            self.resume()
            is_stored = True
        else:
            self._logger.warning("VAD manager buffer is null")
        self._logger.debug(f"VAD result stored with result: {is_stored}")
        return is_stored

    def _notify_result(self, selection_result: CVADSelectionResult, last_index: bool):
        """
        Notifies VAD selection result to listeners.

        Args:
            selection_result (CVADSelectionResult): VAD selection result.
            last_index: Last index.
        """
        self._logger.debug("Notifying VAD result")
        for listener in self._listeners:
            if listener is not None:
                if selection_result.has_vad:
                    listener.on_vad(selection_result.speech_index)

                if selection_result.has_sd:
                    listener.on_sd(selection_result.silence_index)

                listener.on_processed_chunk(selection_result.chunks_and_speech_status,
                                            last_index)
            else:
                self._logger.warning("VAD result: listener is null")

    def _notify_end(self):
        """
        Notifies end to listeners.
        """
        self._logger.debug("Notifying VAD end")
        self._vad_selection_manager.reset()
        for listener in self._listeners:
            listener.on_end_vad()

    def _notify_error(self):
        """
        Notifies error to listeners.
        """
        self._logger.warning("Notifying VAD error")
        for listener in self._listeners:
            listener.on_error_vad()

    def set_last_index(self, id: int):
        """
        Set last id when processing from file

        Args:
            id: last id

        """
        if self._last_id != id:
            self._last_id = id
            self._logger.info(f"Index {id} has marked as the last one")

    def _run(self):
        """
        The method that will be executed by the thread
        """
        self._logger.log_begin("_run")
        self._logger.debug(f"{self.get_thread_name()}: Thread started")
        self._init_thread()

        while not self._finish_thread.is_set():
            try:
                self.suspend()
                while not self._finish_thread.is_set() and not self._buffer.is_empty():
                    try:
                        if self._mode == "AudioChunks":
                            frontal_audio_chunk = self._buffer.get_item()
                            self._logger.debug(f"VAD notify: getting chunk {frontal_audio_chunk.id}")

                            vad_inference = self._vad_recognizer.inference(
                                frames=frontal_audio_chunk.buffer, return_seconds=False)
                            vad_result = CVADResult(
                                vad_inference[0], frontal_audio_chunk.id, round(vad_inference[1], 4))
                        elif self._mode == "Labels":
                            vad_result = self._buffer.get_item()
                            self._logger.debug("VAD notify: getting vad result"
                                                     f" {vad_result.id}")

                        if vad_result.id == self._last_id:
                           vad_result.is_last_index = True

                        self._logger.debug(f"Sending VAD result to selection algorithm "
                                                 f"{vad_result.id}")
                        results_vad_selection = self._vad_selection_manager.process_vad_results_from_selection(
                            vad_result)
                        results_vad_selection = CVADSelectionResult(
                            chunks_and_speech_status=results_vad_selection[
                                "m_chunks_and_speech_status"],
                            has_vad=results_vad_selection["has_vad"],
                            has_sd=results_vad_selection["has_sd"],
                            speech_index=results_vad_selection["m_vad_index"],
                            silence_index=results_vad_selection["m_sd_index"],
                            id=vad_result.id)
                        try:
                            self._notify_result(results_vad_selection, vad_result.is_last_index)
                            if vad_result.is_last_index:
                                self._notify_end()
                                break
                        except Exception as ex:
                            self._logger.error(
                                f"{self._thread_name}: An error occurred while notifying: {ex}")
                            self._notify_error()
                    except Exception as ex:
                        self._logger.error(f"{self._thread_name}: An error occurred while doing the task: {ex}")
                        self._notify_error()
                        self.end_thread()
            except Exception as ex:
                self._logger.error(f"{self._thread_name}: Something went wrong in the "
                                         f"working thread: {ex}")
                self._notify_error()
                self.end_thread()

        self._logger.log_end("_run")

    def is_active(self) -> bool:
        """
        Check is active

        Returns:
            bool: True if the vad manager is active, False otherwise.
        """
        return not self._buffer.is_empty() and self._is_active

    def has_pending_items(self) -> bool:
        """
        Check is the buffer has pending items

        Returns:
            bool: True if the buffer is not empty, False otherwise.
        """
        if self._buffer is not None:
            result = not self._buffer.is_empty()
        else:
            result = False
        return result

    def _signal_handler(self, signum, frame):
        """
        Signal handler

        Args:
            signum:
            frame:
        """
        self._logger.warning("The signal handler has been invoked")
        self._notify_end()
        self.stop()
