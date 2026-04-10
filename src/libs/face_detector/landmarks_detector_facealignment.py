# -*- coding: utf-8 -*-
import numpy as np
import platform
from typing import List
import torch
from face_alignment import FaceAlignment, LandmarksType

from .landmarks_detector_abstract import ALandMarksDetector
# DEFINITIONS AND MACROS
NUM_LANDMARKS = 68


class CLandmarksDetectorFaceAlignment(ALandMarksDetector):
    """
    LandMarks detector for face aligment

    Args:
        cfg_dict: configuration dictionary
    """

    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        # TODO: GPU version not working in Windows
        self.__device = 'cuda' if (torch.cuda.is_available() and platform.system() != "Windows") \
            else 'cpu'
        # models can be dlib, sfd or blazeface
        self.__face_aligner = FaceAlignment(LandmarksType.TWO_D, face_detector='sfd',
                                              device=self.__device)
        self._logger.set_new_name("Landmarks_detector_FaceAlignment")
        self._logger.info("Model Face Alignment detection loaded successfully")

    def _compute_landmarks(self, frame: np.ndarray, bbox_rectangle: list) -> list:
        """
        Convert faces to landmarks

        Args:
            frame: original image
            bbox_rectangle: faces detected with its confidence

        Returns:
            Face landmarks
        """
        self._logger.dev("Computing face alignment landmarks")
        landmarks = self.__face_aligner.get_landmarks(frame, detected_faces=[
            bbox_rectangle])
        self._logger.dev("Face aligment landmarks computed successfully")
        if landmarks is None or len(landmarks) == 0:
            landmarks = np.array([])
        else:
            landmarks = self.__landmark_to_np(landmarks[0])
        return landmarks

    @staticmethod
    def __landmark_to_np(landmarks_in: List) -> np.ndarray:
        """
        Transform landmark to numpy

        Args:
            landmarks_in: List landmarks

        Returns:
            landmarks in numpy format.
        """
        landmark_arr = np.zeros((NUM_LANDMARKS, 2), dtype=object)
        for i in range(NUM_LANDMARKS):
            if landmarks_in[i] is not None and len(landmarks_in[i]) == 2:
                landmark_arr[i] = (int(landmarks_in[i][0]), int(landmarks_in[i][1]))
            else:
                landmark_arr[i] = (None, None)

        return landmark_arr
