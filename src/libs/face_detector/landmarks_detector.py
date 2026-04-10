# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))

from typing import Dict
import numpy as np
from omegaconf import DictConfig, OmegaConf
import importlib

from src.libs.logger.log import getLogger


class CLandmarksDetector:
    """Factory for landmark detectors.

    Dynamically loads the appropriate landmark detector implementation
    (DLib, FaceAlignment, or MediaPipe) based on the configuration file.
    """

    def __init__(self, cfg_dict: Dict | DictConfig = None):
        self._logger = getLogger(self.__class__.__name__)

        if cfg_dict is None:
            raise ValueError("cfg_dict cannot be None")
        elif isinstance(cfg_dict, DictConfig):
            self.config = cfg_dict
        elif isinstance(cfg_dict, dict):
            self.config = OmegaConf.create(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict or omegaconf.DictConfig")

    def __new__(cls, cfg_dict: Dict | DictConfig = None):
        if cfg_dict is None:
            raise ValueError("cfg_dict cannot be None")
        elif isinstance(cfg_dict, DictConfig):
            config = cfg_dict
        elif isinstance(cfg_dict, dict):
            config = OmegaConf.create(cfg_dict)
        else:
            raise TypeError("cfg_dict must be a dict or omegaconf.DictConfig")

        model_name = config.get("model_name", None)
        if model_name is None:
            raise ValueError("model_name must be defined in landmarks config")

        # Map configuration name → module file
        module_map = {
            "DLib": "landmarks_detector_dlib",
            "FaceAlignment": "landmarks_detector_facealignment",
            "MediaPipe": "landmarks_detector_mediapipe",
            "FaceRecognition": "landmarks_detector_facerecognition",
        }

        module_name = module_map.get(model_name)
        if module_name is None:
            raise ValueError(f"Invalid landmarks detector model: {model_name}")

        try:
            # Dynamically import ONLY the selected module
            module = importlib.import_module(f".{module_name}", package=__package__)
            class_name = f"CLandmarksDetector{model_name}"
            model_class = getattr(module, class_name)

            # Return instance of the concrete detector
            return model_class(config)
        except Exception as e:
            raise RuntimeError(f"Failed to load landmarks detector '{model_name}': {e}")
