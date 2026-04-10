# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
import cv2
#disable tensorflow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"
import mediapipe as mp

from .landmarks_detector_abstract import ALandMarksDetector
from src.libs.processing.image_processing.image_processing import CImageProcessing

# DEFINITIONS AND MACROS
NUM_LANDMARKS = 68


class CLandmarksDetectorMediaPipe(ALandMarksDetector):
    """
    LandMarks detector for MediaPipe

    Args:
        cfg_dict: configuration dictionary
    """
    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        self._mediapipe_to_68_landmarks = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
            71, 63, 105, 66, 107, 336, 296, 334, 293, 301, 168, 197, 5, 4, 75, 97, 2, 326, 305,
            33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380, 61, 39, 37, 0, 267, 269,
            291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.__face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                        refine_landmarks=True,
                                                        min_detection_confidence=cfg_dict.threshold)

        self._logger.set_new_name("Landmarks_detector_MediaPipe")
        self._logger.info("Model MediaPipe Face detection loaded successfully")

    def _compute_landmarks(self, frame: np.ndarray, bbox_rectangle: list) -> list:
        """
        Convert faces to landmarks

        Args:
            frame: original image
            bbox_rectangle: faces detected with its confidence

        Returns:
            Face landmarks
        """
        cropped_face = CImageProcessing.crop_image_bbox2(frame, bbox_rectangle)
        results = self.__face_mesh.process(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            landmarks = np.array([])
        else:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.zeros((NUM_LANDMARKS, 2), dtype=int)
            left, top, right, bottom = bbox_rectangle
            for idx, landmark_idx in enumerate(self._mediapipe_to_68_landmarks):
                x = int(face_landmarks.landmark[landmark_idx].x * (right - left) + left)
                y = int(face_landmarks.landmark[landmark_idx].y * (bottom - top) + top)
                landmarks[idx] = (x, y)
        return landmarks
