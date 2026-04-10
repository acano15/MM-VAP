# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
import math
import cv2
from collections import OrderedDict
from typing import Optional, Tuple
from abc import abstractmethod
import matplotlib.colors as mcolors
from omegaconf import DictConfig, OmegaConf

from .head_pose_estimator import HeadPoseEstimator
from src.libs.data.facial_landmarks import EFacialLandmarks
from src.libs.logger.log import getLogger

# DEFINITIONS
colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in map(
    mcolors.hex2color, list(mcolors.TABLEAU_COLORS.values())[:8])]


class ALandMarksDetector:
    """
    LandMarks detector abstract module

    Args:
        cfg_dict: configuration dictionary
    """

    frame = None
    facial_landmarks_idxs = OrderedDict(
        [("mouth", (EFacialLandmarks.OUTER_LIPS_48,
                    EFacialLandmarks.INNER_LIPS_67)),
         ("right_eyebrow", (EFacialLandmarks.RIGHT_EYEBROW_22,
                            EFacialLandmarks.RIGHT_EYEBROW_26)),
         ("left_eyebrow", (EFacialLandmarks.LEFT_EYEBROW_17,
                           EFacialLandmarks.LEFT_EYEBROW_21)),
         ("right_eye", (EFacialLandmarks.RIGHT_EYE_42,
                        EFacialLandmarks.RIGHT_EYE_47)),
         ("left_eye", (EFacialLandmarks.LEFT_EYE_36,
                       EFacialLandmarks.LEFT_EYE_41)),
         ("nose", (EFacialLandmarks.NOSE_BRIDGE_27,
                   EFacialLandmarks.NOSE_BOTTOM_35)),
         ("jaw", (EFacialLandmarks.JAWLINE_0,
                  EFacialLandmarks.JAWLINE_16)),
         ("chin", (EFacialLandmarks.JAWLINE_7,
                   EFacialLandmarks.JAWLINE_11))])

    def __init__(self, cfg_dict: dict=None):
        self._logger = getLogger(self.__class__.__name__)
        self.__head_pose_estimator = HeadPoseEstimator()


    @abstractmethod
    def _compute_landmarks(self, frame: np.ndarray) -> np.ndarray:
        pass

    def inference(self, frame: np.ndarray, face_bboxes: list, relative: bool = True) -> list:
        """
        Convert faces to landmarks

        Args:
            frame: original image
            face_bboxes: faces detected with its confidence
            relative: if True, then not global coordinates are given

        Returns:
            Face landmarks
        """
        faces_landmarks = []
        for bbox_dict in face_bboxes:
            bbox = bbox_dict['bbox']
            landmarks_dict = {"ID": bbox_dict['ID']}
            try:
                landmarks = self._compute_landmarks(frame, bbox)
            except Exception as e:
                landmarks = np.array([])
                self._logger.error(f"Error when computing landmarks: {e}")

            landmarks_dict["global_landmarks"] = landmarks
            landmarks_arr_relative = landmarks.copy()
            if relative:
                for i, (x, y) in enumerate(landmarks):
                    if x is not None and y is not None:
                        landmarks_arr_relative[i][0] = max(x - bbox[0], 0)
                        landmarks_arr_relative[i][1] = max(y - bbox[1], 0)
                    else:
                        landmarks_arr_relative[i] = (None, None)
            landmarks_dict["relative_landmarks"] = landmarks_arr_relative
            landmarks_dict["pose_estimation"] = self.__get_pose_estimation(landmarks.astype(int),
                                                                           frame.shape[1],
                                                                           frame.shape[0])
            faces_landmarks.append(landmarks_dict)

        return faces_landmarks

    def draw_landmarks(
            self, frame: np.ndarray, img_landmarks: dict, is_relative: bool = True) -> np.ndarray:
        """
        Draw the facial landmarks for multiple faces

        Args:
            frame: original image
            img_landmarks: image landmarks
            is_relative: relative

        Returns:
            image drawn.
        """
        image = frame.copy()

        for landmarks_dict in img_landmarks:
            if is_relative:
                landmarks = landmarks_dict['relative_landmarks']
            else:
                landmarks = landmarks_dict['global_landmarks']

            image = self.draw_landmark(image, landmarks)
            image = self.draw_head_pose(image, landmarks_dict["pose_estimation"], landmarks)
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
        image = frame.copy()

        for i, (pos_x, pos_y) in enumerate(landmarks):
            cv2.circle(image, (pos_x, pos_y), 1, (0, 0, 0), -1)

        for idx, (key, val) in enumerate(self.facial_landmarks_idxs.items()):
            points = np.array([landmarks[val[0]:val[1] + 1]], np.int32)
            fill_poly = bool(key in ("right_eye", "left_eye", "mouth"))
            cv2.polylines(image, points, fill_poly, colors[idx], thickness=1)

        return image

    def draw_head_pose(self, frame: np.ndarray, head_pose: Tuple[float, float, float],
                       landmarks: np.ndarray, size: float = 50) -> np.ndarray:
        """
        Draw the head pose (pitch, yaw, roll) on the image using arrows.

        Args:
            frame: The image to draw on.
            head_pose: The head pose angles (pitch, yaw, roll).
            landmarks: Array of 68 landmarks.
            size: The size of the arrows indicating the head pose.

        Returns:
            The image with the pose drawn on it.
        """
        image = frame.copy()
        pitch, yaw, roll = head_pose

        if pitch is not None:
            FONT_SCALE = 1e-3
            THICKNESS_SCALE = 3e-3
            image = frame.copy()
            frame_width = image.shape[1]
            frame_height = image.shape[0]
            font_scale = min(frame_width, frame_height) * FONT_SCALE
            thickness = math.ceil(min(frame_width, frame_height) * THICKNESS_SCALE)

            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)
            roll = np.deg2rad(roll)

            nose_center = landmarks[29]
            arrow_length = np.linalg.norm(max(landmarks[:, 1]) - min(landmarks[:, 1]))

            axis = np.float32(
                [[arrow_length, 0, 0], [0, -arrow_length, 0], [0, 0, -arrow_length]]).reshape(-1, 3)
            nose_end_points_2D, _ = cv2.projectPoints(
                axis, np.array([pitch, yaw, roll]), np.zeros((3, 1)), np.eye(3), np.zeros((4, 1)))
            nose_end_points_2D = np.int32(nose_end_points_2D).reshape(-1, 2)

            min_x, min_y = np.min(landmarks, axis=0)
            max_x, max_y = np.max(landmarks, axis=0)
            square_size = max(max_x - min_x, max_y - min_y)
            scale_x = square_size / max(abs(nose_end_points_2D[:, 0]))
            scale_y = square_size / max(abs(nose_end_points_2D[:, 1]))
            scaled_end_points = np.zeros_like(nose_end_points_2D)
            scaled_end_points[:, 0] = nose_center[0] + (nose_end_points_2D[:, 0]) * scale_x
            scaled_end_points[:, 1] = nose_center[1] + (nose_end_points_2D[:, 1]) * scale_y
            image = cv2.arrowedLine(
                image, tuple(nose_center), (
                    int(scaled_end_points[0][0]),
                    int(scaled_end_points[0][1])
                    ), (0, 255, 0), thickness
                )
            cv2.putText(
                image, f"Pitch: {np.rad2deg(pitch):.1f}", (nose_center[0], nose_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

            image = cv2.arrowedLine(
                image, tuple(nose_center), (
                    int(scaled_end_points[1][0]),
                    int(scaled_end_points[1][1])
                    ), (139, 0, 0), thickness
                )
            cv2.putText(
                image, f"Yaw: {np.rad2deg(yaw):.1f}", (nose_center[0], nose_center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (139, 0, 0), thickness, cv2.LINE_AA)

            image = cv2.arrowedLine(
                image, tuple(nose_center), (
                    int(scaled_end_points[2][0]),
                    int(scaled_end_points[2][1])
                    ), (0, 0, 255), thickness
                )
            cv2.putText(
                image, f"Roll: {np.rad2deg(roll):.1f}", (nose_center[0], nose_center[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        return image

    def __get_pose_estimation(self, landmarks: np.ndarray, image_width: int, image_height: int) -> \
            Tuple[float, float, float]:
        """
        Get head pose estimation from landmarks

        Args:
            landmarks: face landmarks
            image_width: width of the image
            image_height: height of the image

        Returns:
            Tuple containing pitch, yaw, and roll
        """
        try:
            pitch, yaw, roll = self.__head_pose_estimator(landmarks, image_width, image_height)
        except BaseException as e:
            self._logger.warning(f"Head pose estimation Fails: {e}")
            pitch, yaw, roll = None, None, None
        return (pitch, yaw, roll)
