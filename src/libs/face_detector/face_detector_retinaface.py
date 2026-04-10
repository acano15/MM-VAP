# -*- coding: utf-8 -*-
import numpy as np
from retinaface.pre_trained_models import get_model
from typing import List

from .face_detector_abstract import AFaceDetector


class CFaceDetectorRetinaFace(AFaceDetector):
    """
    Face detector module for RetinaFace

    Args:
        cfg_dict: configuration dictionary
    """
    
    def __init__(self, cfg_dict: dict=None):
        super().__init__(cfg_dict)
        self.__model = get_model("resnet50_2020-07-20", max_size=2048)
        self.__model.eval()
        self._logger.set_new_name("CFaceDetector_RetinaFace")
        self._logger.info("Model RetinaFace Face detection loaded successfully")

    def _get_faces_bbox(self, a_frame: np.ndarray) -> List:
        """
        Recognize all the faces over a threshold.

        Args:
            a_frame: OpenCV image to recognize the face.

        Returns:
            List of faces bounding boxes.
        """
        frame_height = a_frame.shape[0]
        frame_width = a_frame.shape[1]

        try:
            detections = self.__model.predict_jsons(a_frame)
        except Exception as e:
            detections = None
            self._logger.error(f"Error when computing retina face prediction: {e}")

        bboxes = []
        if detections is not None and len(detections) > 0:
            for detection in detections:
                confidence = detection['score']
                if confidence > self._threshold:
                    bbox_dict = self.__get_face_from_detection(detection, frame_width, frame_height)
                    face_id = self._face_tracker.get_face_id(a_frame, bbox_dict['bbox'])
                    bbox_dict["ID"] = face_id
                    bboxes.append(bbox_dict)

        if len(bboxes) > 0:
            bboxes = sorted(bboxes, key=lambda k: k['conf'], reverse=True)

        return bboxes

    def __get_face_from_detection(self, a_detection: dict,
                                  a_frame_width: int, a_frame_height: int) -> dict:
        """
        Get single face from detection

        Args:
            a_detection: detection after overpassing the confidence
            a_frame_width: original face width
            a_frame_height: original face height

        Returns:
            dict of bounding box
        """
        bbox_dict = {}
        bbox_dict['conf'] = a_detection['score']
        x1 = int(a_detection['bbox'][0])
        y1 = int(a_detection['bbox'][1])
        x2 = int(a_detection['bbox'][2])
        y2 = int(a_detection['bbox'][3])
        bbox_dict['bbox'] = [x1, y1, x2, y2]
        bbox_dict['width'] = max(x2 - x1, y2 - y1)
        return bbox_dict