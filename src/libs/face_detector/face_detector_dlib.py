# -*- coding: utf-8 -*-
import numpy as np
import cv2
import dlib
from typing import List

from .face_detector_abstract import AFaceDetector


class CFaceDetectorDLib(AFaceDetector):
    """
    Face detector module using Dlib HOG + SVM detector

    Args:
        threshold: not used, maintained for compatibility
    """

    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        self.__face_detector = dlib.get_frontal_face_detector()
        self._logger.set_new_name("CFaceDetector_Dlib")
        self._logger.info("Model Dlib Face detection loaded successfully")

    def _get_faces_bbox(self, frame: np.ndarray) -> List:
        """
        Recognize all the faces using Dlib

        Args:
            frame: opencv image to recognize the face

        Returns:
            list of faces bounding boxes
        """

        gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.__face_detector(gray)

        bboxes = []
        for det in detections:
            x1 = det.left()
            y1 = det.top()
            x2 = det.right()
            y2 = det.bottom()
            bbox_dict = {
                'conf': 1.0,
                'bbox': [x1, y1, x2, y2],
                'width': x2 - x1,
                'height': y2 - y1,
            }
            bbox_dict["area"] = bbox_dict['width'] * bbox_dict['height']
            bbox_dict['distance'] = self._compute_face_distance(bbox_dict['bbox'],
                                                                image_size=(frame.shape[1],
                                                                              frame.shape[0]))
            face_id = self._face_tracker.get_face_id(frame, bbox_dict['bbox'])
            bbox_dict["ID"] = face_id
            bboxes.append(bbox_dict)

        return bboxes
