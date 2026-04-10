# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import dlib

from .landmarks_detector_abstract import ALandMarksDetector

# DEFINITIONS AND MACROS
NUM_LANDMARKS = 68


class CLandmarksDetectorDLib(ALandMarksDetector):
    """
    LandMarks detector for DLib

    Args:
        cfg_dict: configuration dictionary
    """
    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        face_predictor_weights = str(Path(cfg_dict.models_folder) / "shape_predictor_68_face_landmarks.dat")
        self.__face_landmark_predictor = dlib.shape_predictor(face_predictor_weights)
        self._logger.set_new_name("Landmarks_detector_DLib")
        self._logger.info("Model DLib detection loaded successfully")

    def _compute_landmarks(self, frame: np.ndarray, bbox_rectangle: list) -> list:
        """
        Convert faces to landmarks

        Args:
            frame: original image
            bbox_rectangle: faces detected with its confidence

        Returns:
            Face landmarks
        """
        bbox_rectangle = dlib.rectangle(
            bbox_rectangle[0], bbox_rectangle[1], bbox_rectangle[2], bbox_rectangle[3])
        landmarks = self.__face_landmark_predictor(frame, bbox_rectangle)
        landmarks = self.__landmark_to_np(landmarks)
        return landmarks

    @staticmethod
    def __landmark_to_np(landmarks_in: list) -> np.ndarray:
        """
        Transform landmark to numpy

        Args:
            landmarks_in: DLib landmarks

        Returns:
            landmarks in numpy format.
        """
        landmark_arr = np.zeros((NUM_LANDMARKS, 2), dtype=object)
        for i in range(NUM_LANDMARKS):
            if landmarks_in and i < landmarks_in.num_parts:
                part = landmarks_in.part(i)
                landmark_arr[i] = (part.x, part.y)
            else:
                landmark_arr[i] = (None, None)

        return landmark_arr
