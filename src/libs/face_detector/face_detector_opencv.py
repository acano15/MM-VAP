# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import cv2
from typing import List

from .face_detector_abstract import AFaceDetector


class CFaceDetectorOpenCV(AFaceDetector):
    """
    Face detector module for OpenCV

    Args:
        cfg_dict: configuration dictionary
    """

    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        face_proto = str(
            (Path(cfg_dict.models_folder) / "opencv_face_detector.pbtxt").resolve(strict=True))
        face_model = str(
            (Path(cfg_dict.models_folder) / "opencv_face_detector_uint8.pb").resolve(strict=True))
        self._logger.debug(f"Loading OpenCV model from {face_model} and {face_proto}")
        self.__faceNet = cv2.dnn.readNet(face_model, face_proto)
        self._logger.set_new_name("CFaceDetector_OpenCV")
        self._logger.info("Model OpenCV Face detection loaded successfully")

    def _get_faces_bbox(self, frame: np.ndarray) -> List:
        """
        Recognize all the faces over a threshold

        Args:
            frame: opencv image to recognize the face

        Returns:
            list of faces bounding boxes
        """
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

        self.__faceNet.setInput(blob)
        try:
            detections = self.__faceNet.forward()
        except Exception as e:
            detections = None
            self._logger.error(f"Error when computing opencv face prediction: {e}")

        bboxes = []
        if detections is not None:
            for i in range(detections.shape[2]):
                detection = detections[0, 0, i]
                confidence = detection[2]
                if confidence > self._threshold:
                    bbox_dict = self.__get_face_from_detection(detection, frame_width, frame_height)
                    face_id = self._face_tracker.get_face_id(frame, bbox_dict['bbox'])
                    bbox_dict["ID"] = face_id
                    bboxes.append(bbox_dict)

        return bboxes

    def __get_face_from_detection(self, detection: list, frame_width: int,
                                  frame_height: int) -> dict:
        """
        Get single faces from detection

        Args:
            detection: detection after overpassing the confidence
            frame_width: original face width
            frame_height: original face height

        Returns:
            list of bounding boxes dict.
        """
        bbox_dict = {}
        bbox_dict['conf'] = detection[2]
        x1 = int(detection[3] * frame_width)
        y1 = int(detection[4] * frame_height)
        x2 = int(detection[5] * frame_width)
        y2 = int(detection[6] * frame_height)
        bbox_dict['bbox'] = [x1, y1, x2, y2]
        bbox_dict['width'] = x2 - x1
        bbox_dict['height'] = y2 - y1
        bbox_dict["area"] = bbox_dict['width'] * bbox_dict['height']
        bbox_dict['distance'] = self._compute_face_distance(bbox_dict['bbox'],
                                                            image_size=(frame_width,
                                                                          frame_height))
        return bbox_dict
