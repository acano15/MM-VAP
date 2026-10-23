# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/01/2023
# version = '0.2'
# ---------------------------------------------------------------------------
"""

import os
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import numpy as np
import math
import cv2
import av
import time
import copy
from datetime import datetime
from typing import Any, List
from threading import Thread
import threading
import asyncio

from src.libs.counter_per_second.counter_per_second import CCountsPerSec
from src.libs.thread.thread_abstract import AThread
from src.libs.processing.image_processing.image_processing import CImageProcessing
from src.libs.data.image_data.image_format import CImageFormat
from src.libs.data.image_data.image_data import CImageData
from src.libs.data.image_data.image_buffer import CCircularImageBuffer
from src.libs.keyboard_interaction.keyboard_interaction import keyboard, keyboard_handler
from src.libs.logger.log import getLogger
from src.libs.utils.utilities import threaded_method, threaded_method_join

# DEFINITIONS AND MACROS
IMAGES = "images"
VIDEO = "video"
CAMERA = "camera"
VIRTUAL_CAMERA = "virtual_camera"
FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONT_COLOR = (76, 155, 78)
FONT_SCALE = 1e-3
THICKNESS_SCALE = 3e-4
TEXT_Y_OFFSET_SCALE = 2e-2


class CCaptureReader(AThread):
    """
    Capture reader module

    Args:
        source: select camera source. 0 by default
        threaded: boolean that activate the threading mode
        format: image format
        show_video: boolean that activate show video
        resize_per: percentage to resize the original image
        save_video: true to save video
        save_interval: seconds to save the video chunks. Default 60secs
        fps_per: percentage to resize the original fps
        write_fps: used to write the fps into the image
        write_date: used to write the date into the image
        video_path: path to save the video output
    """

    def __init__(self, source: int = 0, format: CImageFormat = None, threaded: bool = True,
                 show_video: bool = True, resize_per: int = 100, save_video: bool = False,
                 save_interval: int = 60, fps_per: int = 100, write_fps: bool = False,
                 write_date: bool = False, video_path: str = None):
        self._cap = None
        self._source = source
        self._threaded = threaded
        self._format = format
        self._show_video = show_video
        self._resize_per = resize_per
        self._save_video = save_video
        self._save_interval = save_interval
        self._fps_per = fps_per
        self._write_fps = write_fps
        self._write_date = write_date
        self._listeners = []
        self._is_stopped = True
        self._images_source = None
        if os.path.isfile(self._source):
            self._source_name = os.path.splitext(os.path.basename(self._source))[0]
        else:
            self._source_name = str(self._source).split('@')[-1].replace(
                '/', '_').replace(":", "_")

        self._instance_name = f"CVideoReader_{self._source_name}"
        super().__init__(f"Thread_{self._instance_name}")

        self._window_name = self._source_name + "_frames"
        self._fps = None
        self._video_active = False
        self._counter = 0
        self._mtx = threading.Lock()
        self._video_condition = threading.Condition(self._mtx)
        self._image_processing = CImageProcessing()
        self._buffer = None
        self._framespersecond = CCountsPerSec(f"CounterVideo_{self._source_name}")
        self._frames_total = -1
        self._has_fps_adjustment = False

        self._logger = getLogger(self._instance_name)

        self._prepare_source()
        if not threaded:
            self._buffer_length = 2
        else:
            self._buffer_length = int(self._fps) * 2

        if self._save_video:
            if video_path is None:
                video_path = os.path.normpath(os.getcwd())
            else:
                video_path = os.path.normpath(video_path)
            self._video_path = os.path.join(os.path.normpath(video_path), self._source_name, '')
            os.makedirs(self._video_path, exist_ok=True)
            self._create_video = True
            self._fourcc = cv2.VideoWriter_fourcc(*'MJPG')

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
            self._buffer = CCircularImageBuffer(length=self._buffer_length,
                                                  buffer_name=self._instance_name,
                                                  is_streaming=self._threaded,
                                                  is_replacing=self._threaded)
            self._video_active = True
            self.start_thread()
            Thread(target=lambda: asyncio.run(self.consume_frames()),
                   name=f"{self._thread_name}_consumer").start()
            self._is_stopped = False
            self._framespersecond.start()
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
            self._framespersecond.stop()
            if self._video_active:
                self._stop_stream()

            with self._video_condition:
                self._video_condition.notify_all()

            self._video_active = False
            del self._cap
            self._listeners.clear()
            self.resume()
            if self._buffer is not None:
                self._buffer.stop()

            self.end_thread()
            self._logger.info(f"{self._thread_name} stopped")
        self._logger.log_end("stop")

    def get_fps(self) -> float:
        """
        Get frames per second
        """
        return self._fps

    def get_source_name(self) -> str:
        """
        Get source name
        """
        return self._source_name

    def add_listener(self, listener: Any) -> bool:
        """
        Adds a listener.

        Args:
            listener (IVADListener): listener to be added.

        Returns:
            bool: True if listener is added successfully.
        """
        self._logger.debug("Adding new listener to Video manager")
        self._listeners.append(listener)
        self._logger.debug(
            f"New listener added to Video manager. Total number of listeners: "
            f"{len(self._listeners)}")
        return True

    def _prepare_source(self):
        """
        Prepare Video source
        """
        try:
            if isinstance(self._source, (int, str)):
                if os.path.isdir(self._source):
                    self._images_source = IMAGES
                    files = [os.path.join(self._source, x) for x in os.listdir(self._source)
                             if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if files:
                        self._cap = self.__images_generator(files)
                        self._frames_total = len(files)
                        self._fps = 1
                        self._threaded = False
                        self._logger.info(
                            f"Reading images from the folder: {self._source}")
                    else:
                        raise ValueError(f"No images in source folder {self._source}")
                else:
                    cv2.setUseOptimized(False)  # Disable OpenCV optimizations
                    self._cap = cv2.VideoCapture(self._source)
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if not self._cap.isOpened():
                        self._logger.error(f"Error opening video source {self._source}")
                        raise ValueError(f"Error opening video source {self._source}")

                    if self._format:
                        self._fps = self._format.fps
                    else:
                        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
                        if self._fps == 0:
                            self._source = "OBS Virtual Camera"
                            self._cap = av.open(f"video={self._source}", format="dshow")
                            self._virtual_cam_generator = self.__virutal_cam_generator()
                            self._logger.warning("Camera reports 0 FPS. Calculating FPS "
                                                       "manually assuming Virtual Camera")
                            self._fps = self.__calculate_fps()
                            #TODO: remove this
                            self._fps = 15
                            self._images_source = VIRTUAL_CAMERA
                            self._logger.info(
                                f"Opening virtual camera {self._source} with fps {self._fps}")

                    self._fps = round(self._fps_per * (self._fps / 100), 2)
                    if self._fps != self._cap.get(cv2.CAP_PROP_FPS):
                        self._has_fps_adjustment = True

                    if isinstance(self._source, str) and any(self._source.lower().endswith(
                            ext) for ext in ['.avi', '.mp4']):
                        self._images_source = VIDEO
                        self._frames_total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        self._logger.info(
                            f"Video file {self._source} loaded with fps {self._fps}. "
                            f"Duration {round(self._frames_total / self._fps, 2)} seconds  with "
                            f"{self._frames_total} frames")
                        self._threaded = False
                    elif self._images_source != VIRTUAL_CAMERA:
                        self._images_source = CAMERA
                        self._logger.info(f"Opening camera {self._source} with fps {self._fps}")
            else:
                self._logger.warning(f"Input file not supported {self._source}")
                raise ValueError("Input file not supported")
        except Exception as e:
            self.stop()
            self._logger.error(e)
            raise e

    def _notify_frame(self, frame: CImageData, is_last_file_frame: bool = False):
        """
        Notifies VAD selection result to listeners.

        Args:
            frame (CImageData): Image frame.
            is_last_file_chunk: Indicate the last frame
        """
        for listener in self._listeners:
            listener.notify_frame(frame, is_last_file_frame)

    def _run(self):
        """
        The method that will be executed by the thread
        """
        self._logger.log_begin("_run")
        self._logger.debug(f"{self._thread_name}: Thread started")
        self._init_thread()
        while not self._finish_thread.is_set():
            try:
                self.suspend()
                while not self._finish_thread.is_set() and not self._buffer.is_empty():
                    try:
                        frontal_image = self._buffer.get_item()
                        is_last_frame = False
                        if frontal_image.id == (self._frames_total - 1):
                            self._logger.info(f"Last frame marked at index {frontal_image.id}")
                            is_last_frame = True
                            self._buffer.stop()
                            self._video_active = False

                        self._notify_frame(frontal_image, is_last_frame)
                    except Exception as ex:
                        self._logger.error(f"{self._thread_name}: An error occurred while doing the task: {ex}")
                        self.end_thread()

                with self._video_condition:
                    self._video_condition.notify_all()
            except Exception as ex:
                self._logger.error(f"{self._thread_name}: Something went wrong in the "
                                         f"working thread: {ex}")
                self.end_thread()

        self._logger.log_end("_run")

    def is_active(self) -> bool:
        """
        Check is active

        Returns:
            bool: True if the video is active, False otherwise.
        """
        return self._video_active

    def has_pending_frames(self) -> bool:
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

    async def consume_frames(self):
        """
        Get video images from source
        """
        time_end = 0
        start_time = time.time()
        time_saving = time.time()

        set_false_active = lambda: setattr(self, '_is_active', False)
        if keyboard is not None:
            keyboard_handler.register(keyboard.KeyCode(char='q'), set_false_active)
            keyboard_handler.register(keyboard.Key.esc, self.close_all_videos)
        while self._video_active:
            frame = self._update_frame()
            if frame is not None and self._video_active:
                image = CImageData(frame=copy.deepcopy(frame), id=self._counter)
                self._logger.debug(f"Adding image {image.id}")
                self._buffer.add_item(image)
                self._counter += 1
                if self._show_video:
                    self._show(image.frame)

                if self._save_video:
                    if (time_end - time_saving) > self._save_interval:
                        self._out.release()
                        self._create_video = True
                        time_saving = time.time()
                    self._store_video(image.frame)

                del frame
            elif self._video_active:
                self._logger.warning(f"Image {self._counter} is None. Not added")
                self._frames_total -= 1
            else:
                self._logger.debug(f"Last image {self._counter} is None")
                self._frames_total -= 1

            self._framespersecond.increment()
            if self._has_fps_adjustment:
                elapsed_time = time.time() - start_time
                ideal_frame_time = 1.0 / self._fps
                actual_fps = self._framespersecond.get_counter()
                actual_frame_time = elapsed_time / actual_fps if actual_fps > 0 else ideal_frame_time
                delay = max(0, ideal_frame_time - actual_frame_time)
                await asyncio.sleep(delay)
                start_time = time.time()

            self.resume()

        if not self._is_stopped and not self._buffer.is_empty():
            with self._mtx:
                self._video_condition.wait()
        self.stop()

    def _update_frame(self) -> np.ndarray:
        """
        Update frame from source

        Returns:
            frame
        """
        frame = None
        if self._images_source == VIDEO or self._images_source == CAMERA:
            self._video_active, frame = self._cap.read()
        elif self._images_source == IMAGES:
            self._video_active, frame = next(self._cap)
        elif self._images_source == VIRTUAL_CAMERA:
            self._video_active, frame = next(self._virtual_cam_generator)

        if frame is not None:
            height, width = frame.shape[:2]
            if ((self._format and (self._format.width and self._format.height)) and
                ((self._format.width != width) or self._format.height != height)):
                self._logger.debug(f"Resized frame from {width}x{height} "
                                         f"to {self._format.width}x{self._format.height}")

                resized_frame = self._image_processing.image_resize_ratio(
                    frame, width=self._format.width,
                    height=self._format.height,
                    inter=cv2.INTER_CUBIC)
                del frame
                frame = resized_frame

            if self._resize_per != 100:
                height, width = frame.shape[:2]
                if self._counter == 0:
                    scale = self._resize_per / 100.0
                    self._original_width = int(width * scale)
                    self._original_height = int(height * scale)

                self._logger.debug(
                    f"Resized frame from {width}x{height} "
                    f"to {self._original_width}x{self._original_height}")
                resized_frame = self._image_processing.image_resize(
                    frame, width=self._original_width, height=self._original_height)
                del frame
                frame = resized_frame
            elif self._resize_per == 100 and self._counter == 0:
                self._original_height = int(frame.shape[0])
                self._original_width = int(frame.shape[1])

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            self._logger.info("Manually closing video source")
            self._video_active = False
            self._stop_stream()
        elif k == 27:
            self._logger.info("Manually closing all sources")
            self._video_active = False
            self.close_all_videos()
        else:
            pass

        # Return a deep copy of the frame to transfer ownership
        return frame.copy() if frame is not None else None

    def _show(self, frame: np.ndarray):
        """
        Show video

        Args:
            frame: image frame
        """
        if self._show_video and self._video_active:
            image = frame.copy()
            if self._write_fps or self._write_date:
                self._logger.debug(f"Writing fps and date into the image")
                image = self.draw_fps(image)
            cv2.imshow(self._window_name, image)

    def _store_video(self, frame: np.ndarray):
        """
        Save video into the video_path folder

        Args:
            frame: image frame
        """
        if self._create_video:
            if self._counter > 1:
                self._out.release()
            video_path = os.path.join(
                self._video_path,
                self._source_name + datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + ".avi")
            self._logger.info(f"Saving video as {video_path} with fps {self._fps} and "
                                    f"{frame.shape[1]}x{frame.shape[0]} resolution")
            self._out = cv2.VideoWriter(video_path, self._fourcc, self._fps,
                                          (frame.shape[1], frame.shape[0]))
            self._create_video = False

        if self._video_active:
            self._out.write(np.uint8(frame))

    def draw_fps(self, image: np.ndarray, fps_draw: float = None):
        """
        Add iterations per second text to lower-left corner of a frame.

        Args:
            image: image to draw fps
            fps_draw: fps to draw
        """
        image = image.copy()
        if fps_draw is None:
            fps_draw = self._framespersecond.get_counter()
        else:
            fps_draw = fps_draw

        if not hasattr(self, '_font_scale') or not hasattr(self, '_thickness'):
            self._font_scale = min(self._original_width, self._original_height) * FONT_SCALE
            self._thickness = math.ceil(min(self._original_width, self._original_height) * THICKNESS_SCALE)

        y_position = int(self._original_height * TEXT_Y_OFFSET_SCALE)
        if self._write_date:
            date_text = datetime.now().strftime(self._framespersecond.get_date_format())
            image = cv2.putText(image, date_text,
                                (int(self._original_width * TEXT_Y_OFFSET_SCALE), y_position), FONT,
                                self._font_scale, FONT_COLOR, self._thickness, cv2.LINE_AA)
            y_position += int(self._original_height * TEXT_Y_OFFSET_SCALE * 1.5)

        if self._write_fps:
            fps_text = f"{fps_draw:.2f} iterations/sec"
            image = cv2.putText(image, fps_text,
                                (int(self._original_width * TEXT_Y_OFFSET_SCALE), y_position), FONT,
                                self._font_scale, FONT_COLOR, self._thickness, cv2.LINE_AA)
        return image

    def _stop_stream(self):
        """
        Release existing video
        """
        try:
            self._video_active = False
            self._logger.debug(f"Stopping video stream {self._instance_name}")
            if self._show_video:
                self._logger.debug(f"Closing source {self._instance_name} video display"
                                         f" {self._window_name}")
                cv2.destroyWindow(self._window_name)
            if self._images_source == VIDEO or self._images_source == CAMERA:
                self._cap.release()
            if self._images_source == VIRTUAL_CAMERA:
                self._cap.close()
            elif self._images_source == IMAGES:
                pass

            if self._save_video:
                self._out.release()

            self._logger.debug(f"Stopped video stream {self._instance_name}")
        except Exception as e:
            self._logger.error(e)

    def close_all_videos(self):
        """
        Close all windows including the current video source
        """
        self._logger.debug("Closing all windows...")
        cv2.destroyAllWindows()

    def __calculate_fps(self) -> float:
        """
        Calculate FPS by counting the number of frames per second.

        Returns:
            float: Estimated FPS.
        """
        start_time = time.time()
        frame_counter = 0

        try:
            for raw_frame in self._cap.decode(video=0):
                # Optional: Add a small delay to simulate real-time frame retrieval
                time.sleep(0.01)
                frame_counter += 1
                if time.time() - start_time >= 1.0:
                    break

        except av.AVError as e:
            self._logger.error(f"Failed to decode frame: {e}")
            raise ValueError(f"Failed to decode frame: {e}")

        return frame_counter

    def __virutal_cam_generator(self) -> bool:
        """
        Create a generator from virtual camera

        Returns:
            boolean video active and image frame.
        """
        try:
            video_active = True
            for raw_frame in self._cap.decode(video=0):
                frame = raw_frame.to_ndarray(format='bgr24')  # Convert to OpenCV-compatible format
                yield video_active, frame

            video_active = False
            yield video_active, np.nan
        except av.AVError as e:
            print(f"Error reading from virtual camera: {e}")
            video_active = False
            yield video_active, np.nan
        finally:
            self._cap.close()

    @staticmethod
    def __images_generator(images: List):
        """
        Create a generator from images paths

        Args:
            images: list images
        Yields:
            boolean video active and image frame.
        """
        images_counter = 0
        video_active = True
        while (images_counter < len(images)):
            frame = cv2.imread(images[images_counter], 3)
            yield video_active, frame
            images_counter += 1
        else:
            video_active = False
            yield video_active, np.nan

    def _signal_handler(self, signum, frame):
        """
        Signal handler

        Args:
            signum:
            frame:
        """
        self._logger.warning("The signal handler has been invoked")
        self.stop()
