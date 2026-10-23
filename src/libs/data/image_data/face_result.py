# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/05/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


from functools import cached_property
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from src.libs.data.data_abstract import AData
from src.libs.data.image_data.image_data import CImageData
from src.libs.processing.vvad_preprocessing import CVVADPreprocessing
from src.libs.logger.log import getLogger

# DEFINITION
LIPS_INDEXES = list(range(48, 68))
NOSECHIN_INDEXES = list(range(2, 15)) + list(range(31, 36))


@dataclass
class CFaceData(AData):
    """
    Face detection data class.

    Args:
        image: original image used for face detection.
        bounding_box: Tuple[int, int, int, int] Bounding box of the detected face (x, y, width, height).
        landmarks: List[Tuple[int, int]] List of facial landmarks (x, y) coordinates.
        confidence: Optional[float] Confidence score of the face detection.
        id: Optional[int] ID of the face detection data.
    """
    __image: CImageData = field(init=False, repr=False)
    __bounding_box: Tuple[int, int, int, int] = field(init=False, repr=False)
    __landmarks: List[Tuple[int, int]] = field(init=False, repr=False)
    __confidence: Optional[float] = field(init=False, repr=False)

    def __init__(self, image: CImageData, bounding_box: Tuple[int, int, int, int],
                 landmarks: List[Tuple[int, int]], confidence: Optional[float] = None,
                 id: Optional[int] = None):
        if id is None:
            face_id = image.id
        else:
            face_id = id

        super().__init__(face_id)
        self._logger = getLogger(f"FaceData_{self._id}")

        self.__processing = CVVADPreprocessing()
        self.__original_image = image
        self.__bounding_box = bounding_box
        self.__landmarks = landmarks
        self.__confidence = confidence
        self.__face_image = self.__crop_face

        self._logger.dev(
            f"Face detection data created for image ID {self._id} with bounding box {self.__bounding_box}")

    @property
    def original_image(self) -> CImageData:
        """
        Get the image ID used for face detection.

        Returns:
            CImageData: Original image.
        """
        original_image = self.__original_image
        self._logger.trace(f"Accessing image: {original_image.id}")
        return original_image

    @property
    def face_image(self) -> CImageData:
        """
        Get the detected face image.

        Returns:
            CImageData: The face image if detected.
        """
        self._logger.trace(f"Accessing face image with id: {self._id}")
        return self.__face_image

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of the detected face.

        Returns:
            Tuple[int, int, int, int]: Bounding box of the face (x, y, width, height).
        """
        self._logger.trace(f"Accessing bounding_box: {self.__bounding_box}")
        return self.__bounding_box

    @property
    def landmarks(self) -> List[Tuple[int, int]]:
        """
        Get the facial landmarks.

        Returns:
            List[Tuple[int, int]]: List of facial landmarks (x, y) coordinates.
        """
        self._logger.trace(f"Accessing landmarks: {self.__landmarks}")
        return self.__landmarks

    @property
    def confidence(self) -> Optional[float]:
        """
        Get the confidence score of the face detection.

        Returns:
            Optional[float]: Confidence score of the face detection.
        """
        self._logger.trace(f"Accessing confidence: {self.__confidence}")
        return self.__confidence

    @cached_property
    def __crop_face(self):
        """
        Crop face image
        """
        face_image = self.__processing.img_processing.crop_image_bbox2(
            self.original_image.frame, self.__bounding_box)

        return CImageData(face_image, self.original_image.id)

    def get_data(self) -> dict:
        """
        Get the data.

        Returns:
            Any: The data.
        """
        return self.__face_image.frame

    def get_roi_image(self, face_part: str = "down_nose_to_chin", mode: str = "relative") -> CImageData:
        """
        Returns the region of interest (RoI) based on the selected face part.

        Returns:
            CImageData:: The RoI image.
        """
        if mode == "relative":
            image = self.__face_image.frame
            landmarks = self.__landmarks["relative_landmarks"]
        elif mode == "global":
            image = self.original_image.frame
            landmarks = self.__landmarks["global_landmarks"]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        if face_part == "down_nose_to_chin":
            roi_image = self.__processing.get_nosechin_region_from_face(image, landmarks)
        else:
            raise ValueError(f"Unsupported face part: {face_part}")
        return CImageData(roi_image, self.original_image.id)

    def copy_face_data(self):
        """
        Creates a custom copy of the CFaceData object, focusing only on copying the image data
        and other necessary attributes, while avoiding unnecessary references to original objects.

        Returns:
            CFaceData: A new instance of CFaceData with copied image data and bounding box.
        """
        copied_image = self.__original_image.copy_image()
        return CFaceData(
            image=copied_image,
            bounding_box=self.__bounding_box,
            landmarks=self.__landmarks,
            id=self._id
        )

    def set_new_id(self, id: int):
        """
        Set new ID for the face data

        Args:
            id (int): New ID
        """
        self._logger.debug(f"Setting new ID: {id}")
        self._id = id

    def __getstate__(self):
        """
        Control what gets pickled.
        Exclude the logger, since it cannot be pickled.
        """
        state = self.__dict__.copy()
        if '_CFaceData__original_image' in state:
            state['_CFaceData__original_image'] = self.original_image
        if '_logger' in state:
            del state['_logger']
        if '_CFaceData__processing' in state:
            del state['_CFaceData__processing']
        return state

    def __setstate__(self, state):
        """
        Restore the state and reinitialize the logger.
        """
        original_image = state.get('_CFaceData__original_image')
        state['_CFaceData__original_image'] = original_image
        self.__dict__.update(state)
        self._logger = getLogger(f"FaceData_{self._id}")
