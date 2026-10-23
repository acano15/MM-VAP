# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 23/05/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
from typing import Any, List, Tuple
from omegaconf import DictConfig

from .face_detector import CFaceDetector
from .landmarks_detector import CLandmarksDetector
from src.libs.data.image_data.image_buffer import CCircularImageBuffer
from src.libs.data.image_data.image_data import CImageData
from src.libs.thread.thread_abstract import AThread
from src.libs.logger.log import getLogger

# DEFINITIONS AND MACROS
BUFFER_LENGTH = 25


class CFaceDetectorManager(AThread):
    """
    Face detector Manager

    Args:
        cfg_dict: recognition configuration with face_detector and landmarks_detector sections.
    """

    def __init__(self, cfg_dict: DictConfig):
        thread_name = "CFaceManager"
        self._buffer = None
        self._listeners = []
        self._is_stopped = True
        self._is_active = True
        super().__init__(f"Thread_{thread_name}")
        self._logger = getLogger(thread_name)

        self._last_id = -1
        self._face_detector = CFaceDetector(cfg_dict.face_detector)
        self._landmark_detector = CLandmarksDetector(cfg_dict.landmarks_detector)

        self._buffer = CCircularImageBuffer(
            BUFFER_LENGTH, buffer_name=self._thread_name,
            is_streaming=False, is_replacing=False)

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

            self._face_detector.reset_tracker()
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
        self._buffer.reset()
        self.start()
        self._logger.log_end("reset")

    def add_listener(self, listener: Any) -> bool:
        """
        Adds a listener.

        Args:
            listener (IfaceListener): Listener to be added.

        Returns:
            bool: True if listener is added successfully.
        """
        self._logger.debug("Adding new listener to Face manager")
        self._listeners.append(listener)
        self._logger.debug(
            f"New listener added to Face manager. Total number of listeners: "
            f"{len(self._listeners)}")
        return True

    def notify_frame(self, frame: CImageData, is_last_file_frame: bool) -> bool:
        """
        Notifies a new frame.

        Args:
            frame (CImageData): Received frame.
            is_last_file_frame: Indicate the last frame

        Returns:
            bool: True if frame is stored successfully, False otherwise.
        """
        original_frame_img = frame.copy_image()
        self._logger.debug(f"Face notify: storing frame {original_frame_img.id}")
        result = False
        if is_last_file_frame:
            self.set_last_index(original_frame_img.id)

        if self._buffer:
            self._buffer.add_item(original_frame_img)
            self.resume()
            result = True
        else:
            self._logger.warning("Face manager buffer is null")
        self._logger.debug(f"frame stored with result: {result}")
        return result

    def _notify_all_result(self, frame: CImageData, faces_bbox: np.ndarray, landmarks: List):
        """
        Notifies face result to listeners.

        Args:
            frame (CImageData): Received frame.
            faces_bbox (CFaceData): face result.
            landmarks: Last index.
        """
        self._logger.debug("Notifying Faces result")
        for listener in self._listeners:
            if listener is not None:
                listener.on_face(frame, faces_bbox, landmarks)
            else:
                self._logger.warning("face result: listener is null")

    def _notify_end(self):
        """
        Notifies end to listeners.
        """
        self._logger.debug("Notifying Face recognition end")
        for listener in self._listeners:
            listener.on_end_face()

    def _notify_error(self):
        """
        Notifies error to listeners.
        """
        self._logger.warning("Notifying error")
        for listener in self._listeners:
            listener.on_error_face()

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
                        frontal_image = self._buffer.get_item()
                        faces_bbox, landmarks = self.get_faces_landmarks_from_image(
                            frontal_image.frame)
                        try:
                            # TODO: Fix for single person notification
                            self._notify_all_result(frontal_image, faces_bbox, landmarks)
                            if frontal_image.id == self._last_id:
                                self._notify_end()
                                break
                        except Exception as ex:
                            self._logger.error(
                                f"{self._thread_name}: An error occurred while notifying: {ex}")
                            self._notify_error()
                        del frontal_image
                    except Exception as ex:
                        self._logger.error(f"{self._thread_name}: An error occurred while doing the task: {ex}")
                        self.end_thread()
            except Exception as ex:
                self._logger.error(f"{self._thread_name}: Something went wrong in the "
                                         f"working thread: {ex}")
                self.end_thread()

        self._logger.log_end("_run")

    def get_faces_landmarks_from_image(self, frame: np.ndarray) -> Tuple[dict, dict]:
        """
        Recognize all the faces over a threshold

        Args:
            frame: opencv image to recognize the face

        Returns:
            Tuple of faces bounding boxes and landmarks
        """
        self._logger.log_begin("get_faces_landmarks_from_image")
        faces_bbox = self._face_detector.inference(frame)
        landmarks = self._landmark_detector.inference(frame, faces_bbox, relative=True)
        self._logger.log_end("get_faces_landmarks_from_image")
        return faces_bbox, landmarks

    def get_faces_from_bboxes(self, frame: np.ndarray, face_bboxes: list) -> List[np.ndarray]:
        """
        Get faces from original image

        Args:
            frame: original image
            face_bboxes: faces detected with its confidence

        Returns:
            List of faces.
        """
        faces = self._face_detector.get_faces(frame, face_bboxes)
        return faces

    def draw_bboxes(self,
                    frame: np.ndarray, bboxes: list, show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes

        Args:
            frame: original image
            bboxes: faces detected with its confidence
            show_confidence: True if we want to show the confidence

        Returns:
            image drawn.
        """

        image = self._face_detector.draw_bboxes(
            frame=frame, bboxes=bboxes, show_confidence=show_confidence)
        return image

    def draw_bbox(self, frame: np.ndarray, bbox: dict, show_confidence: bool = True,
                  color: tuple = (255, 0, 0)) -> np.ndarray:
        """
        Draw a single bounding box

        Args:
            frame: original image
            bbox: face detected with its confidence
            show_confidence: True if we want to show the confidence
            color: Bounding box and text color (default is red: (255, 0, 0))

        Returns:
            image drawn.
        """

        image = self._face_detector.draw_bbox(
            frame=frame, bbox=bbox, show_confidence=show_confidence, color=color)
        return image

    def draw_landmarks(self, frame: np.ndarray, img_landmarks: dict, is_relative) -> np.ndarray:
        """
        Draw the facial landmarks 68

        Args:
            frame: original image
            img_landmarks: image landmarks
            is_relative: relative

        Returns:
            image drawn.
        """
        image = self._landmark_detector.draw_landmarks(
            frame=frame, img_landmarks=img_landmarks, is_relative=is_relative)
        return image

    def draw_landmark(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw facial landmarks for a single face

        Args:
            frame: original image
            landmarks: landmarks for a single face

        Returns:
            image drawn.
        """
        image = self._landmark_detector.draw_landmark(
            frame=frame, landmarks=landmarks)
        return image

    def is_active(self) -> bool:
        """
        Check is active

        Returns:
            bool: True if the vad manager is active, False otherwise.
        """
        return not self._buffer.is_empty() and self._is_active

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
