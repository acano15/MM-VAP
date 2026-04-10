# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
import itertools
import face_recognition

from src.libs.logger.log import getLogger


class CFaceTracker:
    """
    Face detector abstract module
    """
    def __init__(self):
        self._id_generator = itertools.count(1)
        self._known_face_encodings = []
        self._known_face_ids = []

        self._logger = getLogger(self.__class__.__name__)

    def reset(self):
        """
        Reset the face tracker.
        """
        self._id_generator = itertools.count(1)
        self._known_face_encodings = []
        self._known_face_ids = []
        self._logger.debug("Face tracker reset")

    def get_face_id(self, a_frame: np.ndarray, a_face_bbox: tuple) -> int:
        """
        Get the face id from the face encodings

        Args:
            a_frame: opencv image to recognize the face
            a_face_bbox: Tuple (left, top, right, bottom) defining the face bounding box.

        Returns:
            int: Face ID.
        """
        face_id = None
        # Change bbox format to (top, right, bottom, left)
        left, top, right, bottom = a_face_bbox
        face_bbox = (top, right, bottom, left)
        face_encodings = face_recognition.face_encodings(a_frame, [face_bbox])
        if not face_encodings:
            self._logger.error("No face encodings found for the provided bounding box")
        else:
            face_encoding = face_encodings[0]

            if len(self._known_face_ids) == 0:
                face_id = self._register_face(face_encoding)
            else:
                matches = face_recognition.compare_faces(
                    self._known_face_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    face_id = self._known_face_ids[match_index]
                    self._logger.info(f"Recognized face with ID: {face_id}")
                else:
                    face_id = self._register_face(face_encoding)

        return face_id

    def _register_face(self, a_face_encoding: np.ndarray) -> int:
        """
        Register a new face and assign a unique ID.

        Args:
            a_face_encoding: Face encoding to register.

        Returns:
            int: Assigned face ID.
        """
        new_id = next(self._id_generator)
        self._known_face_encodings.append(a_face_encoding)
        self._known_face_ids.append(new_id)
        self._logger.info(f"Registered new face with ID: {new_id}")
        return new_id
