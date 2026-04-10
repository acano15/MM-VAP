# -*- coding: utf-8 -*-
import numpy as np
from typing import Tuple, List
import face_recognition

from .face_detector_abstract import AFaceDetector


class CFaceDetectorFaceRecognition(AFaceDetector):
    """
    Face detector module for faceRecognition

    Args:
        cfg_dict: configuration dictionary
    """

    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        self._logger.set_new_name("CFaceDetector_recognition")
        self._logger.info("Model faceRecognition Face detection loaded successfully")

    def _get_faces_bbox(self, frame: np.ndarray) -> List:
        """
        Recognize all the faces over a threshold

        Args:
            frame: opencv image to recognize the face

        Returns:
            list of faces bounding boxes
        """
        try:
            face_locations = face_recognition.face_locations(frame)
        except Exception as e:
            face_locations = []
            self._logger.error(f"Error when computing face recognition: {e}")

        bboxes = []
        for i, (top, right, bottom, left) in enumerate(face_locations):
            confidence = None
            bbox_dict = self.__get_face_from_detection(top, right, bottom, left, confidence, frame.shape)
            face_id = self._face_tracker.get_face_id(frame, bbox_dict['bbox'])
            bbox_dict["ID"] = face_id
            bbox_dict["area"] = (right - left) * (bottom - top)
            bboxes.append(bbox_dict)

        return bboxes

    def __get_face_from_detection(
        self, top: int, right: int, bottom: int, left: int, confidence: float,
        image_size: Tuple) -> dict:
        """
        Get single faces from detection

        Args:
            top: top coordinate of the face bounding box
            right: right coordinate of the face bounding box
            bottom: bottom coordinate of the face bounding box
            left: left coordinate of the face bounding box
            confidence: confidence score of the detection
            image_size: image original shape

        Returns:
            dict: Bounding box information including distance from camera.
        """
        bbox_dict = {}
        bbox_dict['conf'] = confidence
        bbox_dict['bbox'] = [left, top, right, bottom]
        bbox_dict['width'] = right - left
        bbox_dict['height'] = bottom - top
        bbox_dict['distance'] = self._compute_face_distance(bbox_dict['bbox'], image_size=image_size)
        return bbox_dict
