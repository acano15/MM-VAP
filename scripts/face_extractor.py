# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_path))
sys.path = list(dict.fromkeys(sys.path))

from typing import List, Any
from tqdm import tqdm
import numpy as np
from moviepy.editor import VideoFileClip
import dlib
import cv2
import gc
from multiprocessing import Lock
from omegaconf import DictConfig, OmegaConf

from src.libs.data.face_result import CFaceData
from src.libs.processing import CImageProcessing
from src.libs.logger.log import getLogger

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


class CFaceExtractor:
    def __init__(self, face_detector: Any, landmarks_detector: Any, src_paths: List[str],
                 tgt_dir: Path, tgt_fps: float, tgt_size: tuple,
                 tmp_dir: Path, lock: Lock, available_positions: List):
        self._logger = getLogger(self.__class__.__name__)

        self.face_detector = face_detector
        self.landmarks_detector = landmarks_detector
        self.src_paths = src_paths
        self.tgt_dir = tgt_dir
        self.tgt_fps = tgt_fps
        self.tgt_size = tgt_size
        self.tmp_dir = tmp_dir
        self.lock = lock
        self.available_positions = available_positions

        self.image_processing = CImageProcessing()

    def extract_face_from_video(self, i):
        self.face_detector.reset_tracker()

        src_path = self.src_paths[i]
        separated = str(src_path.absolute()).split(os.sep)
        file_name = src_path.stem
        file_id = separated[-2]
        tgt_path = self.tgt_dir / '{}-{}'.format(file_id, file_name + '.npy')
        if not os.path.exists(tgt_path):
            self._logger.info(f'[{i}] Processing {src_path} -> {tgt_path}')

            tqdm_position = i
            with self.lock:
                for position_candidate, available in enumerate(self.available_positions):
                    if available:
                        tqdm_position = position_candidate
                        self.available_positions[tqdm_position] = False
                        break

            with (self.lock):
                cap = cv2.VideoCapture(str(src_path.absolute()))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                if (self.tgt_fps is not None) and (fps != self.tgt_fps):
                    cap.release()
                    clip = VideoFileClip(str(src_path.absolute()))
                    new_src_path = self.tmp_dir / src_path.name
                    clip.write_videofile(
                        str(new_src_path.absolute()), fps=self.tgt_fps, verbose=False, logger=None)
                    clip.reader.close()
                    clip.audio = None  # Release audio resources
                    del clip
                    src_path = new_src_path
                    cap = cv2.VideoCapture(str(src_path.absolute()))

                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                no_face_cnt = 0
                processed_frames = np.zeros(
                    (
                        num_frames, self.tgt_size[0], self.tgt_size[1], 3), dtype=np.uint8)

                self._logger.debug(f"Processing video {src_path} with {num_frames} frames")
                pbar = tqdm(
                    total=num_frames, position=tqdm_position + 1,
                    desc='Row {:02d} - index {:02d}: Processing {}'.format(
                        tqdm_position + 1, i, src_path.absolute()),
                    leave=None)

                for j in range(num_frames):
                    pbar.update(1)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    missing = False
                    faces_bbox = self.face_detector.inference(frame)
                    num_faces = len(faces_bbox)
                    if num_faces != 0:
                        self._logger.dev(f"Found {num_faces} faces in frame {j}")
                        landmarks_detected = self.landmarks_detector.inference(
                            frame, faces_bbox, relative=True)
                        if landmarks_detected:
                            face_bbox = faces_bbox[0]["bbox"]
                            landmarks_face = landmarks_detected[0]
                            face_data = CFaceData(j, frame, face_bbox, landmarks_face)
                            face_image = face_data.face_image
                            face_image = self.image_processing.image_resize(
                                face_image, width=self.tgt_size[0], height=self.tgt_size[1],
                                inter=cv2.INTER_CUBIC)
                        else:
                            missing = True
                            self._logger.dev(
                                f"No landmarks detected in frame {j}, skipping face extraction")
                            no_face_cnt += 1
                    else:
                        missing = True
                        self._logger.dev(f"No faces detected in frame {j}")
                        no_face_cnt += 1

                    if missing:
                        face_image = np.zeros((self.tgt_size[0], self.tgt_size[1], 3),
                                              dtype=np.uint8)

                processed_frames[j] = face_image

                pbar.close()
                cap.release()

                if len(processed_frames) == 0:
                    self._logger.error(
                            f'No faces detected in {src_path}. {tgt_path} will not be '
                            'created')
                else:
                    processed_frames = np.stack(processed_frames).astype(np.uint8)
                    self._logger.info(f"[{i}] Saving processed frames to {tgt_path}")
                    np.save(str(tgt_path), processed_frames)

                gc.collect()

            with self.lock:
                self.available_positions[tqdm_position] = True
        else:
            self._logger.info(f"[{i}] {tgt_path} already exists, skipping extraction")

        return None

    def safe_detect(self, detector, frame, index):
        try:
            # Ensure it's a NumPy array
            if not isinstance(frame, np.ndarray):
                self._logger.warning(f"[{index}] Frame is not a numpy array: {type(frame)}")

            if frame.dtype != np.uint8:
                self._logger.warning(f"[{index}] Converting dtype {frame.dtype} to uint8")
                frame = frame.astype(np.uint8)

            h, w, c = frame.shape
            if len(frame.shape) == 2 or len(frame.shape) == 3:
                pass
            elif len(frame.shape) == 3 and h > w:
                self._logger.debug(f"[{index}] Converting BGR to RGB")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"[{index}] Unsupported frame shape: {frame.shape}")

            dets = detector(frame, 0)
            return dets
        except Exception as e:
            self._logger.warning(f"[{index}] Face detection failed: {e}")
            return []
