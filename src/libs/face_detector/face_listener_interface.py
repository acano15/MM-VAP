# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 18/06/2024
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List

class IFaceListener(ABC):
    """
    Interface for Face Listener
    """

    @abstractmethod
    def on_face(self, frame: Any, faces_bbox: np.ndarray, landmarks: List):
        """
        Called when Face manager detects faces.

        Args:
            frame (Any): Received frame.
            faces_bbox: face result.
            landmarks: Last index.
        """
        pass

    @abstractmethod
    def on_end_face(self):
        """
        Called when Face ends.
        """
        pass

    @abstractmethod
    def on_error_face(self):
        """
        Called when Face errors.
        """
        pass
