# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/01/2023
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import numpy as np
import time
import wave
from datetime import datetime
from typing import Any, List
from threading import Thread
import threading

from src.libs.counter_per_second.counter_per_second import CCountsPerSec
from src.libs.thread.thread_abstract import AThread
from src.libs.processing.audio_processing.audio_processing import CAudioProcessing
from src.libs.data.audio_data.audio_format import CAudioFormat
from src.libs.data.audio_data.audio_buffer import CCircularAudioBuffer
from src.libs.data.audio_data.audio_chunk import CAudioChunk
from src.libs.keyboard_interaction.keyboard_interaction import keyboard, keyboard_handler
from src.libs.logger.log import getLogger
# DEFINITIONS AND MACROS
AUDIO_FILE = "audio_file"
MICROPHONE = "microphone"


class CAudioReader(AThread):
    """
    Audio reader module

    Args:
        source: microphone source
        chunk_size: length of extraction
        format: audio format
        threaded: boolean that activate the threading mode
        show_signal: boolean that activate show signal
        resample_per: percentage to resize the original sample rate
        save_audio: boolean that indicates to save the audio
        save_interval: time in seconds to save audio chunks
        audio_path: Audio path to save the audios
    """

    def __init__(self, source: int, chunk_size: int, format: CAudioFormat,
                 threaded: bool = True, show_signal: bool = False, resample_per: int = 100,
                 save_audio: bool = False, save_interval: int = 60, audio_path: str = None):

        self._source = str(source) if isinstance(source, os.PathLike) else source
        self._chunk_size = chunk_size
        self._format = format
        self._threaded = threaded
        self._show_signal = show_signal
        self._resize_per = resample_per
        self._save_audio = save_audio
        self._save_interval = save_interval
        self._listeners = []
        self._is_stopped = True

        if os.path.isfile(self._source):
            self._source_name = os.path.splitext(os.path.basename(self._source))[0]
        else:
            self._source_name = f"Microphone_{self._source}"

        self._instance_name = f"CAudioReader_{self._source_name}"
        super().__init__(f"Thread_{self._instance_name}")

        self._window_name = os.path.join(self._source_name, "chunks")
        self._sr = self._format.sample_rate
        self._audio_signal = None
        self._audio_active = False
        self._counter = 0
        self._mtx = threading.Lock()
        self._audio_condition = threading.Condition(self._mtx)
        self._initial_condition = threading.Condition()
        self._audio_processing = CAudioProcessing()
        self._buffer = None
        self._chunkspersecond = CCountsPerSec(f"CounterAudio_{self._source_name}")
        self._audio_source = ""
        self._chunks_total = -1

        self._logger = getLogger(self._instance_name)

        self._prepare_source()
        if not threaded:
            self._buffer_length = int(self._sr / self._chunk_size) #2
        else:
            self._buffer_length = int(self._sr / self._chunk_size)

        if self._save_audio:
            if audio_path is None:
                audio_path = os.path.normpath(os.getcwd())
            else:
                audio_path = os.path.normpath(audio_path)
            self._audio_path = os.path.join(audio_path, self._source_name, '')
            os.makedirs(self._audio_path, exist_ok=True)
            self._create_audio = True

    def __del__(self):
        self.stop()
        super().__del__()

    def start(self):
        """
        Start
        """
        self._logger.log_begin("start")
        if self._is_stopped:
            self._logger.info(f"Starting {self._thread_name}")
            self._counter = 0
            self._buffer = CCircularAudioBuffer(length=self._buffer_length,
                                                  buffer_name=self._instance_name,
                                                  is_streaming=self._threaded,
                                                  is_replacing=self._threaded)
            self._initialized = False
            self._audio_active = True
            self.start_thread()
            self._is_stopped = False
            if self._audio_source == MICROPHONE:
                self._start_stream()

            Thread(target=self.consume_chunks, name=f"{self._instance_name}_consumer").start()
            self._chunkspersecond.start()
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
            self._is_stopped = True
            self._chunkspersecond.stop()
            if self._audio_active:
                self._stop_stream()

            with self._audio_condition:
                self._audio_condition.notify_all()

            self._audio_active = False
            self._listeners.clear()
            self.resume()
            if self._buffer is not None:
                self._buffer.stop()

            self.end_thread()
            if self._show_signal:
                try:
                    import matplotlib.pyplot as plt
                    plt.close()
                except RuntimeError as e:
                    self._logger.error(f"Error while closing {self._thread_name}: {e}")

            self._logger.info(f"{self._thread_name} stopped")
        self._logger.log_end("stop")

    def add_listener(self, listener: Any) -> bool:
        """
        Adds a listener.

        Args:
            listener (IVADListener): listener to be added.

        Returns:
            bool: True if listener is added successfully.
        """
        self._logger.debug("Adding new listener to Audio manager")
        self._listeners.append(listener)
        self._logger.debug(
            f"New listener added to Audio manager. Total number of listeners: "
            f"{len(self._listeners)}")
        return True

    def _prepare_source(self):
        """
        Prepare Audio source
        """
        try:
            if isinstance(self._source, str):
                self._load_audio()
                self._audio_source = AUDIO_FILE
                self._threaded = False
            elif isinstance(self._source, int):
                try:
                    import pyaudio
                except ImportError as exception:
                    raise ImportError(
                        "PyAudio is required for microphone input") from exception
                self._p = pyaudio.PyAudio()
                self.__check_microphone_existence()
                self._audio_source = MICROPHONE
            else:
                self._logger.warning(f"Input file not supported {self._source}")
                raise ValueError("Input file not supported")
        except Exception as e:
            self.stop()
            self._logger.error(e)
            raise e

    def _start_stream(self):
        """
        Start microphone streaming
        """
        self._logger.log_begin("_start_stream")
        try:
            self._logger.info(f"Starting microphone audio streaming {self._source_name}")
            default_sr = int(self._p.get_device_info_by_index(self._source)['defaultSampleRate'])
            if default_sr != self._sr:
                self._logger.warning(f"{self._source_name} sample rate selected {self._sr} "
                                           f"differs from the default {default_sr}")

            self._stream = self._p.open(
                input_device_index=self._source,
                format=self._format.bit_depth,
                channels=self._format.number_channels,
                rate=self._sr,
                input=True,
                output=False,
                frames_per_buffer=1024)
            self._logger.info(f"Microphone audio streaming {self._source_name} started")
        except Exception as e:
            self._logger.error(f"Audio reader error: {e}")
            raise e
        self._logger.log_end("_start_stream")

    def _stop_stream(self):
        """
        Stop microphone streaming
        """
        self._logger.log_begin("_stop_stream")
        try:
            self._logger.debug(f"Stopping audio stream {self._source_name}")
            if self._audio_source == MICROPHONE:
                self._stream.stop_stream()
                self._stream.close()
                self._p.terminate()
                self._logger.dev(f"Audio {self._source_name} streaming has end")

            self._audio_active = False
            self._logger.debug(f"Stopped audio stream {self._source_name}")
        except Exception as e:
            self._logger.error(e)
        self._logger.log_end("_stop_stream")

    def _update_samples(self, chunks_counter: int) -> np.array(float):
        """
        Update samples given a windows length

        Args:
            chunks_counter: chunks counter

        Returns:
            Float numpy array with the audio chunk.
        """
        try:
            init_sample_counter = chunks_counter * self._chunk_size
            end_sample_counter = min((chunks_counter + 1) * self._chunk_size, len(self._audio_signal))
            audio_chunk = self._audio_signal[init_sample_counter: end_sample_counter]
            if len(audio_chunk) < self._chunk_size:
                audio_chunk = np.pad(audio_chunk, (0, self._chunk_size - len(audio_chunk)))
            self._chunkspersecond.increment(self._chunk_size)
            if len(self._audio_signal) == end_sample_counter:
                self._audio_active = False
        except Exception as e:
            self._logger.error(e)
            raise e
        return audio_chunk

    def consume_chunks(self) -> tuple:
        """
        Create a generator from audio samples

        Yields:
            tuple of boolean audio active and CAudio chunk object
        """
        with self._initial_condition:
            if not self._initialized:
                self._logger.debug("Waiting until initialized")
                self._initial_condition.wait()

        time_saving = 0
        self._audio_signal_save = np.empty((0,), dtype=np.float64)
        if keyboard is not None:
            keyboard_handler.register(keyboard.KeyCode(char='q'), self._stop_stream)
            keyboard_handler.register(keyboard.Key.esc, self._stop_stream)
        while self._audio_active:
            audio_buffer = None
            if self._audio_source == AUDIO_FILE:
                audio_buffer = self._update_samples(self._counter)
            elif self._audio_source == MICROPHONE:
                audio_buffer = np.fromstring(self._stream.read(self._chunk_size),
                                             dtype=np.float32)
                self._audio_signal_save = np.concatenate((self._audio_signal_save, audio_buffer))

            if audio_buffer is not None:
                audio_chunk = CAudioChunk(id=self._counter, buffer=audio_buffer)
                self._logger.debug(f"Adding chunk {audio_chunk.id}")
                self._buffer.add_item(audio_chunk)
                self._counter += 1
            elif self._audio_active:
                self._logger.warning(f"Audio buffer {self._counter} is None. Not added")
                self._chunks_total -= 1

            self.resume()

            if self._show_signal:
                self.display_signal(audio_chunk.buffer)
                self.play_signal(audio_chunk.buffer)

            time_end = time.time()
            if self._save_audio:
                if (time_end - time_saving) > self._save_interval:
                    self._create_audio = True
                    time_saving = time.time()
                    self._store_audio()
                    self._audio_signal_save = np.empty((0,), dtype=np.float64)

        if not self._is_stopped and not self._buffer.is_empty():
            with self._mtx:
                self._audio_condition.wait()
        self.stop()

    def _notify_chunk(self, chunk: CAudioChunk, is_last_file_chunk: bool = False):
        """
        Notifies VAD selection result to listeners.

        Args:
            chunk (CAudioChunk): Audio chunk.
            is_last_file_chunk: Indicate the last chunk
        """
        for listener in self._listeners:
            listener.notify_chunk(chunk, is_last_file_chunk)

    def _run(self):
        """
        The method that will be executed by the thread
        """
        self._logger.log_begin("_run")
        self._logger.debug(f"{self._thread_name}: Thread started")
        self._init_thread()
        while not self._finish_thread.is_set():
            try:
                with self._initial_condition:
                    self._initialized = True
                    self._initial_condition.notify_all()

                self.suspend()
                while not self._finish_thread.is_set() and not self._buffer.is_empty():
                    try:
                        frontal_audio_chunk = self._buffer.get_item()
                        is_last_chunk = False
                        if frontal_audio_chunk.id == (self._chunks_total - 1):
                            is_last_chunk = True

                        self._notify_chunk(frontal_audio_chunk, is_last_chunk)
                    except Exception as ex:
                        self._logger.error(f"{self._thread_name}: An error occurred while doing the task: {ex}")
                        self.end_thread()

                with self._audio_condition:
                    self._audio_condition.notify_all()
            except Exception as ex:
                self._logger.error(f"{self._thread_name}: Something went wrong in the working thread: {ex}")
                self.end_thread()

        self._logger.log_end("_run")

    def is_active(self) -> bool:
        """
        Check is active

        Returns:
            bool: True if the audio is active, False otherwise.
        """
        return self._audio_active

    def has_pending_chunks(self) -> bool:
        """
        Check is the buffer has pending chunks

        Returns:
            bool: True if the buffer is not empty, False otherwise.
        """
        if self._buffer is not None:
            result = not self._buffer.is_empty()
        else:
            result = False
        return result

    def get_sample_rate(self) -> float:
        """
        Get sample rate

        Returns:
            Sample rate.
        """
        return self._sr

    def get_audio_signal(self) -> np.array(float):
        """
        Get audio signal

        Returns:
            Audio signal array.
        """
        return self._audio_signal

    def display_signal(self, audio_signal: np.array(float)):
        """
        Display audio signal

        Args:
            audio_signal: Audio signal
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pyformulas as pf
        import librosa

        self._fig = plt.figure("Audio signal")
        screen = pf.screen(title='Audio signal waveform')
        librosa.display.waveshow(audio_signal, sr=self._sr, alpha=0.4, color='C1')
        plt.xlabel('time')
        plt.ylabel('Amplitude')
        plt.title("Audio signal waveform")
        self._fig.canvas.draw()
        image = np.fromstring(self._fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
        screen.update(image)
        plt.clf()

    def play_signal(self, audio_signal: np.array(float)):
        """
        Play audio signal

        Args:
            audio_signal: Audio signal
        """
        Thread(target=self.__play_signal, args=(audio_signal,)).start()

    def _store_audio(self):
        """
        Save audio
        """
        audio_path = os.path.join(
            self._audio_path, datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + ".wav")
        self._logger.info(f"Saving audio as {audio_path}")
        wave_file = wave.open(audio_path, 'wb')
        wave_file.setnchannels(self._format.number_channels)
        if self._audio_source == AUDIO_FILE:
            wave_file.setsampwidth(4)
        elif self._audio_source == MICROPHONE:
            wave_file.setsampwidth(self._p.get_sample_size(self._format.bit_depth))
        wave_file.setframerate(self._format.sample_rate)
        wave_file.writeframes(b''.join(self._audio_signal_save))
        wave_file.close()

    def _load_audio(self):
        """Load audio"""
        try:
            import librosa

            self._audio_signal, sr = librosa.load(self._source, sr=None, mono=True)
            if self._audio_signal.size == 0:
                raise ValueError(f"Audio file is empty: {self._source}")
            if self._sr != sr:
                self._logger.info(f"Original audio signal resample from {sr} to"
                                        f" {self._sr}")
                self._audio_signal = self._audio_processing.resample_audio_signal(
                    self._audio_signal, sr=sr, new_sr=self._sr)

            self._chunks_total = int(np.ceil(len(self._audio_signal) / self._chunk_size))
            self._logger.info(f"Audio file {self._source} loaded with sr {self._sr}. Duration "
                                    f"{round(len(self._audio_signal) / self._sr, 2)} seconds with "
                                    f"{self._chunks_total} chunks")
        except Exception as e:
            self._logger.error(f"Audio reader error: {e}")
            raise e

    def __check_microphone_existence(self):
        """
        Check if the microphone exists
        """
        device_info = self._p.get_device_info_by_index(self._source)
        if device_info is None or device_info.get('maxInputChannels') == 0:
            self._logger.error(f"Audio reader microphone: {self._source} does not exist")
            raise ValueError(f"Audio reader microphone: {self._source} does not exist")

    def __play_signal(self, audio_signal: np.array):
        """
        Play the audio signal using pyaudio.

        Args:
            audio_signal: Audio signal as numpy array
        """
        try:
            import pyaudio
        except ImportError as exception:
            raise ImportError("PyAudio is required for audio playback") from exception

        audio_signal = audio_signal.astype(np.float32).tobytes()

        stream = pyaudio.PyAudio().open(
            format=pyaudio.paFloat32,
            channels=self._format.number_channels,
            rate=self._sr,
            output=True)

        stream.write(audio_signal)
        stream.stop_stream()
        stream.close()

    def _signal_handler(self, signum, frame):
        """
        Signal handler

        Args:
            signum:
            frame:
        """
        self._logger.warning("The signal handler has been invoked")
        self._audio_active = False
        try:
            with self._audio_condition:
                self._audio_condition.notify_all()
            self._stop_stream()
        except Exception as e:
            self._logger.error(e)
        self.stop()
