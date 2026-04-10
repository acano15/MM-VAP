# -*- coding: utf-8 -*-
import numpy as np
from functools import cached_property
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

from src.libs.processing.processing import CPreprocessing
from src.libs.data.facial_landmarks import LIPS_INDEXES, NOSECHIN_INDEXES
from src.libs.logger.log import getLogger


@dataclass
class CFaceData:
    """
    Face detection data class.

    Args:
        id: int ID of the face detection data.
        image: original image used for face detection.
        bounding_box: Tuple[int, int, int, int] Bounding box of the detected face (x, y, width, height).
        landmarks: List[Tuple[int, int]] List of facial landmarks (x, y) coordinates.
        confidence: Optional[float] Confidence score of the face detection.
    """
    _image: np.ndarray = field(default=None, repr=False)
    _bounding_box: Tuple[int, int, int, int] = field(init=False, repr=False)
    _landmarks: List[Tuple[int, int]] = field(init=False, repr=False)
    _confidence: Optional[float] = field(init=False, repr=False)

    def __init__(self, id: Any, image: np.array, bounding_box: Tuple[int, int, int, int],
                 landmarks: List[Tuple[int, int]], confidence: Optional[float] = None):
        if id is None:
            self._id = "None"
        else:
            self._id = id

        self._logger = getLogger(f"FaceData_{self._id}")

        self._processing = CPreprocessing()

        self._original_image = image
        self._bounding_box = bounding_box
        self._landmarks = landmarks
        self._confidence = confidence
        self._face_image = self.__crop_face

        self._logger.dev(
            f"Face detection data created for image ID {self._id} with bounding box {self._bounding_box}")

    @property
    def id(self) -> int:
        """
        Get the data id.

        Returns:
            int: The id of the data.
        """
        return self._id

    @property
    def original_image(self) -> np.array:
        """
        Get the image ID used for face detection.

        Returns:
            np.array: Original image.
        """
        self._logger.trace(f"Accessing original image with id: {self._id}")
        return self._original_image

    @property
    def face_image(self) -> np.array:
        """
        Get the detected face image.

        Returns:
            np.array: The face image if detected.
        """
        self._logger.trace(f"Accessing face image with id: {self._id}")
        return self._face_image

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of the detected face.

        Returns:
            Tuple[int, int, int, int]: Bounding box of the face (x, y, width, height).
        """
        self._logger.trace(f"Accessing bounding_box: {self._bounding_box}")
        return self._bounding_box

    @property
    def landmarks(self) -> List[Tuple[int, int]]:
        """
        Get the facial landmarks.

        Returns:
            List[Tuple[int, int]]: List of facial landmarks (x, y) coordinates.
        """
        self._logger.trace(f"Accessing landmarks: {self._landmarks}")
        return self._landmarks

    @property
    def confidence(self) -> Optional[float]:
        """
        Get the confidence score of the face detection.

        Returns:
            Optional[float]: Confidence score of the face detection.
        """
        self._logger.trace(f"Accessing confidence: {self._confidence}")
        return self._confidence

    @cached_property
    def __crop_face(self):
        """
        Crop face image
        """
        face_image = self._processing.img_processing.crop_image_bbox2(
            self.original_image, self._bounding_box)

        return face_image

    def get_roi_image(self, face_part: str = "down_nose_to_chin", landmarks_type: str = "relative_landmarks") -> np.array:
        """
        Returns the region of interest (RoI) based on the selected face part.

        Returns:
            np.array:: The RoI image.
        """
        if landmarks_type == "relative_landmarks":
            image = self._face_image
            landmarks = self._landmarks["relative_landmarks"]
            landmarks = np.array(landmarks, dtype=np.float32)
        elif landmarks_type == "global_landmarks":
            image = self.original_image
            landmarks = self._landmarks["global_landmarks"]
            landmarks = np.array(landmarks, dtype=np.int16)
        else:
            raise ValueError(f"Unsupported mode: {landmarks_type}")

        if face_part == "down_nose_to_chin":
            roi_image = self._processing.get_nosechin_region_from_face(image, landmarks)
        else:
            raise ValueError(f"Unsupported face part: {face_part}")
        return roi_image

    def copy_face_data(self):
        """
        Creates a custom copy of the CFaceData object, focusing only on copying the image data
        and other necessary attributes, while avoiding unnecessary references to original objects.

        Returns:
            CFaceData: A new instance of CFaceData with copied image data and bounding box.
        """
        copied_image = self._original_image.copy()
        return CFaceData(
            _image=copied_image,
            _bounding_box=self._bounding_box,
            _landmarks=self._landmarks,
            _id=self._id
        )

    def set_new_id(self, _id: int):
        """
        Set new ID for the face data

        Args:
            _id (int): New ID
        """
        self._logger.debug(f"Setting new ID: {_id}")
        self._id = _id

    def __getstate__(self):
        """
        Control what gets pickled.
        Exclude the logger, since it cannot be pickled.
        """
        state = self.__dict__.copy()
        if '_CFaceDat_original_image' in state:
            state['_CFaceDat_original_image'] = self.original_image
        if '_logger' in state:
            del state['_logger']
        if '_CFaceDat_processing' in state:
            del state['_CFaceDat_processing']
        return state

    def __setstate__(self, state):
        """
        Restore the state and reinitialize the logger.
        """
        original_image = state.get('_CFaceDat_original_image')
        state['_CFaceDat_original_image'] = original_image
        self.__dict__.update(state)
        self._logger = getLogger(f"FaceDat_{self._id}")
