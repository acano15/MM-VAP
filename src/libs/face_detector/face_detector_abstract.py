# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
import math
import cv2
from abc import abstractmethod
from omegaconf import DictConfig, OmegaConf

from .face_tracker import CFaceTracker
from src.libs.processing.image_processing.image_processing import CImageProcessing
from src.libs.logger.log import getLogger


class AFaceDetector:
    """
    Face detector abstract module

    Args:
        threshold: threshold to accept the inference
    """

    face_predictor_weights = None
    face_landmark_predictor = None
    frame = None
    image_processing = CImageProcessing()

    def __init__(self, cfg_dict: dict | DictConfig = None):
        self._logger = getLogger("CFaceDetector")

        if cfg_dict is None:
            raise ValueError("cfg_dict cannot be None")
        elif isinstance(cfg_dict, DictConfig):
            self.config = cfg_dict
        elif isinstance(cfg_dict, dict):
            self.config = OmegaConf.create(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict, omegaconf.DictConfig, or None")

        self._threshold = self.config.threshold
        self._face_tracker = CFaceTracker()
        self._logger.info(f"Face Detector configured with threshold {self._threshold}")

    @abstractmethod
    def _get_faces_bbox(self, frame: np.ndarray) -> np.ndarray:
        pass

    def inference(self, frame: np.ndarray) -> np.ndarray:
        """
        Inference the faces from single image

        Args:
            frame: opencv image to recognize the face

        Returns:
            list of faces bounding boxes [[x1, y1, x2, y2]]
        """
        face_bboxes = self._get_faces_bbox(frame)
        face_bboxes = self._filter_by_nms(face_bboxes)
        return face_bboxes

    def reset_tracker(self):
        """
        Reset the face tracker.
        """
        self._face_tracker.reset()

    def get_faces(self, frame: np.ndarray, face_bboxes: list) -> list:
        """
        Get faces from original image

        Args:
            frame: original image
            face_bboxes: faces detected with its confidence

        Returns:
            List of faces.
        """
        faces = []
        for bbox_dict in face_bboxes:
            bbox = bbox_dict['bbox']
            face_cropped = self.image_processing.crop_image_bbox2(frame, bbox)
            faces.append(face_cropped)
        return faces

    def draw_bboxes(
            self,
            frame: np.ndarray, bboxes: list, show_confidence: bool = True) -> np.ndarray:
        """
        Draw bounding boxes for multiple faces

        Args:
            frame: original image
            bboxes: faces detected with its confidence
            show_confidence: True if we want to show the confidence

        Returns:
            image drawn.
        """
        FONT_SCALE = 1e-3
        THICKNESS_SCALE = 3e-3
        TEXT_Y_OFFSET_SCALE = 2e-2
        image = frame.copy()
        frame_width = image.shape[1]
        frame_height = image.shape[0]
        font_scale = min(frame_width, frame_height) * FONT_SCALE
        thickness = math.ceil(min(frame_width, frame_height) * THICKNESS_SCALE)

        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                image = self.draw_bbox(image, bbox, show_confidence=show_confidence)

            if len(bboxes) == 1:
                str_face = ' Face'
            else:
                str_face = ' Faces'
            num_faces = str(len(bboxes)) + str_face + ' Found'
            R = 0
            G = 255
        else:
            num_faces = "No face Detected"
            R = 255
            G = 0

        cv2.putText(
            image, num_faces,
            (int(frame_width * TEXT_Y_OFFSET_SCALE), int(frame_width * TEXT_Y_OFFSET_SCALE)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            thickness=thickness, color=(0, G, R))
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
        FONT_SCALE = 1e-3
        THICKNESS_SCALE = 3e-3
        TEXT_Y_OFFSET_SCALE = 2e-2
        image = frame.copy()
        frame_width = image.shape[1]
        frame_height = image.shape[0]
        font_scale = min(frame_width, frame_height) * FONT_SCALE
        thickness = math.ceil(min(frame_width, frame_height) * THICKNESS_SCALE)

        x1, y1, x2, y2 = bbox['bbox']
        conf = bbox['conf']
        if conf:
            conf = round(conf * 100, 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        face_id = bbox.get('ID', 'Unknown')
        text = f'Face ID: {face_id}'
        if show_confidence:
            text += f' ({conf}%)'

        cv2.putText(
                image, text,
                (x1, y1 - int(frame_height * TEXT_Y_OFFSET_SCALE)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                thickness=thickness, color=color)

        return image

    def _compute_face_distance(self, bbox: list, image_size: tuple) -> float:
        """
        Compute a parameter that indicates the relative size of the face in the image.
        This parameter will be larger when the face is closer to the camera.

        Args:
            bbox: Bounding box coordinates [top, right, bottom, left].
            image_size: Size of the original image as (height, width).

        Returns:
            Normalized face size parameter.
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height
        image_area = image_size[0] * image_size[1]
        face_distance_param = bbox_area / image_area
        return round(face_distance_param, 4)

    def _filter_by_nms(self, bboxes: list, iou_threshold: float = 0.3) -> list:
        """
        Apply Non-Maximum Suppression (NMS) while keeping full dict info.

        Args:
            bboxes: List of detected bounding boxes [{'bbox': [x1, y1, x2, y2], 'conf': float, 'ID': str}]
            iou_threshold: IoU threshold for suppression.

        Returns:
            Filtered list of bounding boxes (same dict format) after NMS.
        """
        if len(bboxes) == 0:
            return bboxes

        # Sort boxes by confidence (descending)
        bboxes = sorted(bboxes, key=lambda x: x["conf"], reverse=True)
        picked = []

        for i, current in enumerate(bboxes):
            keep = True
            for prev in picked:
                iou = self.__iou(current["bbox"], prev["bbox"])
                if iou > iou_threshold:
                    keep = False
                    break
            if keep:
                picked.append(current)

        return picked

    @staticmethod
    def __iou(box1: list, box2: list) -> float:
        """
        Private method to compute the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU: Intersection over Union value between 0 and 1.
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou
