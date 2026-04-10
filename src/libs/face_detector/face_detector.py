# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

from typing import List, Tuple, Optional, Callable
import numpy as np
from omegaconf import DictConfig, OmegaConf
import importlib

from src.libs.logger.log import getLogger


class CFaceDetector:
    """Face detector.

    Attributes:
        cfg_dict (dict): Configuration dictionary.
    """

    def __init__(self, cfg_dict: dict | DictConfig = None):
        self._logger =getLogger(self.__class__.__name__)
        if cfg_dict is None:
            raise ValueError("cfg_dict cannot be None")
        elif isinstance(cfg_dict, DictConfig):
            self.config = cfg_dict
        elif isinstance(cfg_dict, dict):
            self.config = OmegaConf.create(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict, omegaconf.DictConfig, or None")

    def __new__(cls, cfg_dict: dict | DictConfig = None):
        if cfg_dict is None:
            raise ValueError("cfg_dict cannot be None")
        elif isinstance(cfg_dict, DictConfig):
            config = cfg_dict
        elif isinstance(cfg_dict, dict):
            config = OmegaConf.create(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict, omegaconf.DictConfig, or None")

        model_name = cfg_dict.model_name

        # Map configuration name → module file
        module_map = {
            "OpenCV": "face_detector_opencv",
            "DLib": "face_detector_dlib",
            "RetinaFace": "face_detector_retinaface",
            "FaceRecognition": "face_detector_facerecognition",
        }

        module_name = module_map.get(model_name)
        if module_name is None:
            raise ValueError(f"Invalid face detector: {model_name}")

        try:
            # Dynamically import ONLY the selected module
            module = importlib.import_module(f".{module_name}", package=__package__)
            class_name = f"CFaceDetector{model_name}"
            model_class = getattr(module, class_name)

            # Return an instance of the real detector instead of CFaceDetector
            return model_class(config)
        except Exception as e:
            raise e
