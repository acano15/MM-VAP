import numpy as np
import cv2
from typing import List, Any
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from sklearn.preprocessing import minmax_scale

from src.libs.data.facial_landmarks import EFacialLandmarks
from src.libs.processing.image_processing.image_processing import CImageProcessing
from src.libs.processing.audio_processing.audio_processing import CAudioProcessing
from src.libs.logger.log import getLogger


class CPreprocessing:
    """
    Processing module

    Args:
        cfg_dict: Configuration dictionary.
    """

    def __init__(self, cfg_dict: dict | DictConfig = None):
        self._logger = getLogger(self.__class__.__name__)
        self.img_processing = CImageProcessing()
        self.audio_processing = CAudioProcessing()

        if cfg_dict is None:
            self.config = None
        elif isinstance(cfg_dict, DictConfig):
            self.config = cfg_dict
        elif isinstance(cfg_dict, dict):
            self.config = OmegaConf.create(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict, omegaconf.DictConfig, or None")

        if self.config is not None:
            self.image_new_width = self.config.width
            self.image_new_height = self.config.height
            self.channels = self.config.channels

        self._facial_landmarks_idxs = OrderedDict(
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

    def get_nosechin_regions(self, images_buffer: np.ndarray, landmarks: np.ndarray,
                             compute_mean: bool = True) -> np.ndarray:
        """
        Get nosechin region from faces buffer

        Args:
            images_buffer: numpy array with the images of the faces
            landmarks: list of landmarks
            compute_mean: if true apply mean to the data

        Returns:
            numpy array of nosechin regions.
        """
        if compute_mean:
            landmarks = np.moveaxis(
                np.dstack(
                    [np.mean(landmarks, axis=0, dtype=np.float64).astype(int)] * len(
                        images_buffer)),
                -1, 0)
        else:
            landmarks = landmarks

        dtype = images_buffer.dtype
        nosechin_regions = []
        for img, landmark in zip(images_buffer, landmarks):
            if img is not None and landmark is not None:
                if len(landmark) == 0:
                    region = None
                else:
                    region = self.get_nosechin_region_from_face(img, landmark)
                if region is not None and region.dtype != dtype:
                    region = region.astype(dtype)
                nosechin_regions.append(region)
            else:
                nosechin_regions.append(None)

        if not compute_mean:
            result = np.empty(len(nosechin_regions), dtype=dtype)
            for i, nosechin_region in enumerate(nosechin_regions):
                result[i] = nosechin_region
        else:
            result = np.array(nosechin_regions, dtype=dtype)

        return result

    def get_nosechin_region_from_face(self, image: np.ndarray, landmark: np.ndarray) -> np.ndarray:
        """
        Get nosechin region from faces buffer

        Args:
            image: numpy array with the image to compute the ROI
            landmark: face landmarks

        Returns:
            image ROI.
        """
        if landmark is None or len(landmark) == 0:
            return None

        landmark = landmark.copy().astype(np.int32)
        middle_nose_point = landmark[EFacialLandmarks.NOSE_BRIDGE_30]
        down_chin_point = np.maximum.reduce([landmark[EFacialLandmarks.JAWLINE_8],
                                             landmark[EFacialLandmarks.JAWLINE_7],
                                             landmark[EFacialLandmarks.JAWLINE_10],
                                             landmark[EFacialLandmarks.JAWLINE_6],
                                             landmark[EFacialLandmarks.JAWLINE_11]])
        nosechin_height = down_chin_point[1] - middle_nose_point[1]

        left_cheek_point = landmark[EFacialLandmarks.JAWLINE_1]
        right_cheek_point = landmark[EFacialLandmarks.JAWLINE_15]

        if left_cheek_point is not None and right_cheek_point is not None:
            nosechin_width = right_cheek_point[0] - left_cheek_point[0]
            initial_x = middle_nose_point[0] - int(nosechin_width / 2)
        elif left_cheek_point is None and right_cheek_point is not None:
            nosechin_width = (right_cheek_point[0] - middle_nose_point[0]) * 2
            initial_x = middle_nose_point[0] - (right_cheek_point[0] - middle_nose_point[0])
        elif right_cheek_point is None and left_cheek_point is not None:
            nosechin_width = (middle_nose_point[0] - left_cheek_point[0]) * 2
            initial_x = left_cheek_point[0]
        else:
            raise ValueError("Both cheek points are missing")

        initial_x = max(initial_x, 0)

        initial_point = [initial_x, middle_nose_point[1]]
        roi_bbox = [initial_point[0], initial_point[1], initial_point[0] + nosechin_width,
                    initial_point[1] + nosechin_height]

        nosechin_roi_img = self.img_processing.crop_image_bbox2(image, roi_bbox)
        return nosechin_roi_img

    def get_nosechin_regions_landmarks(self, landmarks: np.ndarray, compute_mean: bool = True) \
        -> np.ndarray:
        """
        Get nosechin region from faces buffer

        Args:
            images_buffer: numpy array with the images of the faces
            landmarks: list of landmarks
            compute_mean: if true apply mean to the data

        Returns:
            numpy array of nosechin regions.
        """
        if compute_mean:
            landmarks = np.moveaxis(np.dstack(
                    [np.mean(landmarks, axis=0, dtype=np.float64).astype(int)] * len(
                        landmarks)),
                -1, 0)
        else:
            landmarks = landmarks

        nosechin_landmarks_regions = []
        for landmark in landmarks:
            if landmark is not None:
                region = self.get_nosechin_region_from_landmarks(landmark)
                nosechin_landmarks_regions.append(region)
            else:
                nosechin_landmarks_regions.append(None)

        if not compute_mean:
            result = np.empty(len(nosechin_landmarks_regions), dtype=object)
            for i, nosechin_region in enumerate(nosechin_landmarks_regions):
                result[i] = nosechin_region
        else:
            result = np.array(nosechin_landmarks_regions, dtype=object)

        return result

    def get_nosechin_region_from_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts the nose-chin region from facial landmarks.

        Args:
            landmarks (np.ndarray): Array of facial landmarks (N, 2), where N is the number of landmarks.

        Returns:
            np.ndarray: Subset of landmarks corresponding to the nose-chin region.
        """

        middle_nose_point = landmarks[EFacialLandmarks.NOSE_BRIDGE_30]
        down_chin_point = np.maximum.reduce(
            [landmarks[idx] for idx in [
                EFacialLandmarks.JAWLINE_6, EFacialLandmarks.JAWLINE_7, EFacialLandmarks.JAWLINE_8,
                EFacialLandmarks.JAWLINE_10, EFacialLandmarks.JAWLINE_11]])
        nosechin_height = down_chin_point[1] - middle_nose_point[1]

        left_cheek_point = landmarks[EFacialLandmarks.JAWLINE_1] if 1 < len(landmarks) else None
        right_cheek_point = landmarks[EFacialLandmarks.JAWLINE_15] if 15 < len(
            landmarks) else None

        if left_cheek_point is not None and right_cheek_point is not None:
            nosechin_width = right_cheek_point[0] - left_cheek_point[0]
            initial_x = middle_nose_point[0] - int(nosechin_width / 2)
        elif left_cheek_point is None and right_cheek_point is not None:
            nosechin_width = (right_cheek_point[0] - middle_nose_point[0]) * 2
            initial_x = middle_nose_point[0] - (right_cheek_point[0] - middle_nose_point[0])
        elif right_cheek_point is None and left_cheek_point is not None:
            nosechin_width = (middle_nose_point[0] - left_cheek_point[0]) * 2
            initial_x = left_cheek_point[0]
        else:
            raise ValueError("Both cheek points are missing")

        roi_xin = max(initial_x, 0)
        roi_xax = roi_xin + nosechin_width
        roi_yin = middle_nose_point[1]
        roi_yax = roi_yin + nosechin_height

        roi_landmarks = np.array(
            [
                lm for lm in landmarks
                if roi_xin <= lm[0] <= roi_xax and roi_yin <= lm[1] <= roi_yax
                ])

        if roi_landmarks.shape[1] > 2:
            raise ValueError("Invalid number of landmarks in the nose-chin region.")
        return roi_landmarks

    def fix_sequence_size(self, sequence_data: List, normalize, use_gray=False) -> np.ndarray:
        """
        Fix sequence size in a 4D list

        Args:
            sequence_data: 4D List with variable sequence size
            normalize: Normalize data

        Returns:
            Fix sequence data.
        """
        batch_size = len(sequence_data)
        sequence_size = len(sequence_data[0])
        new_shape = (batch_size, sequence_size, self.image_new_height, self.image_new_width,
                     self.channels)
        dtype = np.float32 if normalize else np.uint8
        sequence_datfix = np.zeros(new_shape, dtype=dtype)
        mask = np.zeros((batch_size, sequence_size), dtype=dtype)
        for idx, data in enumerate(sequence_data):
            sequence_length = len(data)
            for idy, img in enumerate(data):
                if img is None or len(img) == 0:
                    self._logger.warning(f"Zero image found at in sequence {idx} at image {idy}")
                    mask[idx, idy] = 1
                    left_idk = idy - 1
                    right_idk = idy + 1
                    while left_idk >= 0 or right_idk < sequence_length:
                        if left_idk >= 0 and (
                                data[left_idk] is not None and len(data[left_idk]) > 0):
                            img = data[left_idk]
                            break
                        if right_idk < sequence_length and (
                                data[right_idk] is not None and len(data[right_idk]) > 0):
                            img = data[right_idk]
                            break
                        left_idk -= 1
                        right_idk += 1

                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)

                if use_gray:
                    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img[:, :, 0] = gray_image
                    img[:, :, 1] = gray_image
                    img[:, :, 2] = gray_image

                image_resized = self.img_processing.image_resize(img, width=self.image_new_width,
                                                                 height=self.image_new_height,
                                                                 inter=cv2.INTER_CUBIC)
                if normalize:
                    image_resized = self.normalize_data(image_resized, feature_range=(0, 1))

                sequence_datfix[idx, idy] = image_resized

        return sequence_datfix, mask

    def get_nonzeroask(self, sequence_data: np.ndarray) -> np.ndarray:
        """
        Get real data mask from a 4D array

        Args:
            sequence_data: 4D array with variable sequence size

        Returns:
            Mask of non zeros.
        """
        mask = np.zeros((sequence_data.shape[0]))
        for idx, data in enumerate(sequence_data):
            mask[idx] = np.array(sequence_data.shape[1])
            for idy, img in enumerate(data):
                if img is None:
                    mask[idx] = idy
                    break
        return mask

    def draw_landmarks_on_black_images(self, landmarks: np.ndarray, compute_mean: bool = True,
                                       filter: List = []) \
        -> np.ndarray:
        """
        Draw landmarks on a black image using the reference image size and crop only the face region.

        Args:
            landmarks: List of Nx2 landmark arrays (variable size allowed).
            compute_mean: If True, apply mean to the data.
            filter: List of facial landmarks to filter (optional).

        Returns:
            List of images (as np.ndarray) of shape [len(batch), H, W, 3].
        """
        if compute_mean:
            landmarks = np.moveaxis(
                np.dstack(
                    [np.mean(landmarks, axis=0, dtype=np.float64).astype(int)] * len(
                        landmarks)),
                -1, 0)
        else:
            landmarks = landmarks

        landmarks_images = []
        for landmark in landmarks:
            if landmark is not None:
                landmarks_images.append(self.draw_landmarks_on_black_image_single(landmark, filter))
            else:
                landmarks_images.append(np.zeros(
                    (self.image_new_height, self.image_new_width, 3), dtype=np.uint8))

        if not compute_mean:
            result = []
            for i, landmarks_image in enumerate(landmarks_images):
                result.append(landmarks_image)

        result = np.asarray(landmarks_images, dtype=object)
        return result

    def draw_landmarks_on_black_image_single(self, landmark: np.ndarray, filter: List = []) -> np.ndarray:
        """
        Draw landmarks on a black image using the reference image size and crop only the face region.

        Args:
            landmark: Nx2 array of absolute landmark coordinates (can include None).
            filter: List of facial landmarks to filter (optional).

        Returns:
            Cropped black image containing only the face with landmarks.
        """
        padding = 10

        # Convert to object type to preserve None
        landmark = np.array(landmark, dtype=object)

        # Filter valid landmarks for bounding box computation
        valid_landmarks = np.array(
            [
                pt for pt in landmark
                if pt is not None and None not in pt
                ], dtype=np.float64)

        if valid_landmarks.size == 0:
            xin = yin = xax = yax = 0
        else:
            bboxin = np.floor(valid_landmarks.min(axis=0)).astype(np.int64)
            bboxax = np.ceil(valid_landmarks.max(axis=0)).astype(np.int64)
            xin, yin = bboxin
            xax, yax = bboxax

        width = int(xax - xin + 2 * padding)
        height = int(yax - yin + 2 * padding)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        offset = np.array([xin - padding, yin - padding], dtype=np.float64)

        # Shift landmarks and preserve None
        shifted_landmarks = []
        for pt in landmark:
            if pt is None or None in pt:
                shifted_landmarks.append(None)
            else:
                shifted = np.array(pt, dtype=np.float64) - offset
                shifted_landmarks.append(tuple(shifted.astype(int)))

        for i, pt in enumerate(shifted_landmarks):
            if pt is None:
                continue

            x, y = pt
            filtered = False
            if filter:
                for filter_key in filter:
                    start, end = self._facial_landmarks_idxs[filter_key]
                    if not (start <= i <= end):
                        filtered = True
                        break

            if not filtered and (0 <= x < width and 0 <= y < height):
                cv2.circle(image, (x, y), radius=1, color=(255, 255, 255), thickness=-1)

        color = (255, 255, 255)
        if filter:
            valid_regions = {key: self._facial_landmarks_idxs[key] for key in filter}.items()
        else:
            valid_regions = self._facial_landmarks_idxs.items()

        for key, (start, end) in valid_regions:
            if end >= len(shifted_landmarks):
                continue

            region_points = [
                pt for pt in shifted_landmarks[start:end + 1]
                if pt is not None
                ]
            if len(region_points) < 2:
                continue

            points = np.array([region_points], dtype=np.int32)
            is_closed = key in ("right_eye", "left_eye", "mouth")
            cv2.polylines(image, points, is_closed, color, thickness=1)

        return image

    @staticmethod
    def create_landmarksask(image: np.ndarray, landmarks: np.ndarray, landmarks_indexes) -> (np.ndarray):
        """
        Generate a mask around the landmarks of a recognized face with 68 landmarks.

        Args:
            image: The original image.
            landmarks: Array of 68 landmarks.
            landmarks_indexes: Interest landmarks.

        Returns:
            The image with the black mask around the mouth.
        """

        lips_points = landmarks[landmarks_indexes].astype(np.int32)
        hull = cv2.convexHull(lips_points)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, hull, 255)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_inv = cv2.bitwise_not(mask)

        result = cv2.bitwise_and(image, image, mask=mask_inv)

        return result

    @staticmethod
    def normalize_data(data: Any, weights: Any = None, feature_range: Any = (0, 1)) -> np.array:
        """
        Normalize data

        Args:
            data: np.array:
            weights: Any:
            feature_range: Any:

        Returns:
            normalized data np.array.
        """
        shape = data.shape
        datscaled = minmax_scale(data.ravel(), feature_range=feature_range).reshape(shape)
        return datscaled

    @staticmethod
    def denormalize_data(data: np.ndarray, originalin: float, originalax: float,
                         feature_range: tuple = (0, 1)) -> np.ndarray:
        """
        Denormalize data to the original scale.

        Args:
            data: np.ndarray: Normalized data to denormalize.
            originalin: float: Original minimum value of the data before normalization.
            originalax: float: Original maximum value of the data before normalization.
            feature_range: tuple: The range used during normalization (default is (0, 1)).

        Returns:
            np.ndarray: Denormalized data.
        """
        min_range, max_range = feature_range
        shape = data.shape

        datdenormalized = (data - min_range) / (max_range - min_range)
        datdenormalized = datdenormalized * (originalax - originalin) + originalin
        datdenormalized = np.clip(datdenormalized, 0, 255)
        return datdenormalized.reshape(shape)
