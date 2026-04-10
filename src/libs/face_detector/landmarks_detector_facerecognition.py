# -*- coding: utf-8 -*-
import numpy as np
import face_recognition

from .landmarks_detector_abstract import ALandMarksDetector

# DEFINITIONS AND MACROS
NUM_LANDMARKS = 68


class CLandmarksDetectorFaceRecognition(ALandMarksDetector):
    """
    Landmarks detector using the face_recognition library (built on dlib).

    Args:
        cfg_dict: configuration dictionary
    """

    def __init__(self, cfg_dict: dict = None):
        super().__init__(cfg_dict)
        self._logger.set_new_name("Landmarks_detector_FaceRecognition")
        self._logger.info("Model face_recognition landmark detector loaded successfully")

    def _compute_landmarks(self, frame: np.ndarray, bbox_rectangle: list) -> np.ndarray:
        """
        Compute facial landmarks within a given bounding box.

        Args:
            frame: Original BGR image (as numpy array).
            bbox_rectangle: [x1, y1, x2, y2] bounding box of the face.

        Returns:
            np.ndarray: (68, 2) array of landmarks (x, y).
        """
        # Convert BGR → RGB (required by face_recognition)
        rgb_frame = frame[:, :, ::-1]

        # Restrict to the given bounding box
        top, right, bottom, left = bbox_rectangle[1], bbox_rectangle[2], bbox_rectangle[3], bbox_rectangle[0]

        try:
            landmarks_list = face_recognition.face_landmarks(
                rgb_frame,
                face_locations=[(top, right, bottom, left)],
                model="large"  # 68-point model
            )
        except Exception as e:
            self._logger.error(f"Error during face landmark detection: {e}")
            return np.zeros((NUM_LANDMARKS, 2), dtype=object)

        if not landmarks_list:
            self._logger.debug("No landmarks detected.")
            return np.zeros((NUM_LANDMARKS, 2), dtype=object)

        # Convert the dict of keypoints into a consistent 68-point layout
        landmarks = self.__landmarks_to_np(landmarks_list[0])
        return landmarks

    @staticmethod
    def __landmarks_to_np(landmarks_dict: dict) -> np.ndarray:
        """
        Convert face_recognition landmarks dict to numpy array.

        Args:
            landmarks_dict: Dict mapping facial regions (eyes, nose, etc.) to lists of (x, y).

        Returns:
            np.ndarray: (68, 2) array of landmarks (x, y).
        """
        # Flatten all regions into one ordered list
        points = []
        for region_points in landmarks_dict.values():
            points.extend(region_points)

        landmark_arr = np.zeros((NUM_LANDMARKS, 2), dtype=object)
        for i in range(min(len(points), NUM_LANDMARKS)):
            landmark_arr[i] = (points[i][0], points[i][1])
        for i in range(len(points), NUM_LANDMARKS):
            landmark_arr[i] = (None, None)

        return landmark_arr
