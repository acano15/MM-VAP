# -*- coding: utf-8 -*-
"""

pyfeat:
    https://py-feat.org/basic_tutorials/01_basics.html#working-with-multiple-images
    https://github.com/cosanlab/py-feat/blob/main/feat/detector.py
    
dlib python example:
    https://github.com/davisking/dlib/tree/master/python_examples
    
dlib with cuda on conda:
    https://anaconda.org/zeroae/dlib-cuda

"""
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path))
sys.path.insert(0, str((base_path.parent / 'src').resolve()))
sys.path.insert(0, str((base_path.parent / 'src' / 'libs').resolve()))
sys.path.insert(0, str((base_path.parent / 'src' / 'libs' / 'utils').resolve()))
sys.path.insert(0, str((base_path.parent / 'src' / 'libs' / 'logger').resolve()))
sys.path = list(dict.fromkeys(sys.path))

import gc
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import sys
import time
# import torch
from tqdm import tqdm
from pathlib import Path
# import moviepy.editor.VideoFileClip
from moviepy.editor import VideoFileClip
from multiprocessing import Pool, Manager
import dlib
import hydra
from omegaconf import DictConfig

from log import load_logger_config, getLogger
from utils import repo_root, OmegaConf

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def _init_worker(shared_list, logger_cfg):
    global G_EXTRACTOR, G_LOGGER

    load_logger_config(logger_cfg)
    G_LOGGER = getLogger("Main")
    G_EXTRACTOR = Extractor(shared_list)


def _worker_extract(i):
    global G_EXTRACTOR, G_LOGGER
    return G_EXTRACTOR.extract_face_from_video(i, logger=G_LOGGER)


def safe_detect(logger, detector, frame, index):
    try:
        # Ensure it's a NumPy array
        if not isinstance(frame, np.ndarray):
            logger.warning(f"[{index}] Frame is not a numpy array: {type(frame)}")

        if frame.dtype != np.uint8:
            logger.warning(f"[{index}] Converting dtype {frame.dtype} to uint8")
            frame = frame.astype(np.uint8)

        h, w, c = frame.shape
        if len(frame.shape) == 2 or len(frame.shape) == 3:
            pass
        elif len(frame.shape) == 3 and h > w:
            logger.debug(f"[{index}] Converting BGR to RGB")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"[{index}] Unsupported frame shape: {frame.shape}")

        dets = detector(frame, 0)
        return dets
    except Exception as e:
        logger.warning(f"[{index}] Face detection failed: {e}")
        return []




@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig):
    load_logger_config(cfg_dict.logger)
    logger = getLogger("Main")

    OmegaConf.resolve(cfg_dict)

    manager = Manager()
    lock = manager.Lock()

    src_dir = Path("/media/acano/DATA/DBs/MMDS/Audio-Video/HRI_DB/data/subjects/").expanduser().resolve()
    tgt_dir = Path("/media/acano/DATA/DBs/MMDS/Audio-Video/HRI_DB/dataset_extracted/")

    import os
    session_folder =[src_dir / x  for x in os.listdir(src_dir)]
    conversations_folder = [x / y for x in session_folder if x.is_dir() for y in os.listdir(x)]    
    # Unroll the list of conversations
    print(conversations_folder)
    files = [x for x in src_dir.glob('**/*.mp4') if x.is_file() and not x.parent.name[
        0].isdigit() and x.parent.name.startswith(cfg_dict.dataset.dataset_selected.keyword)]
    files =  [x for x in src_dir.glob("*/*/*.mp4")]

    detector_path = os.path.abspath(
        os.path.join(repo_root(), cfg_dict.model.non_verbal_cond.face.detector.face_model_path))

    sp_path = os.path.abspath(
        os.path.join(repo_root(), cfg_dict.model.non_verbal_cond.face.detector.landmark_model_path))

    tgt_size = (112, 112)    
    # tgt_fps = 30
    tgt_fps = None
    tmp_dir = None

    use_multiprocess = False
    use_multiprocess = True
    num_process = 6
    
    if tgt_fps != None:
        
        tmp_dir = Path("{}_{}".format(str(src_dir), tgt_fps))
        if tmp_dir.exists():
            logger.error(f'{str(tmp_dir)} exists')
            sys.exit()
        else:
            tmp_dir.mkdir()

    available_positions = manager.list([True for _ in range(num_process)])
    shared_list = manager.list([detector_path, sp_path, files, tgt_dir, tgt_fps, tgt_size, tmp_dir,
                                lock, available_positions])    
    extractor = Extractor(shared_list)
    
    index_list = list(range(len(files)))
    if use_multiprocess:
        
        tqdm_iterator = TqdmIterator(index_list, position=0, 
                                     desc='{:02d}: Total progress'.format(0), shared_list=shared_list)
        
        logger.debug(f'use multiprocessing with {num_process} processes')
        with Pool(
            processes=num_process,
            initializer=_init_worker,
            initargs=(shared_list, cfg_dict.logger)) as p:
                _ = p.imap(_worker_extract, tqdm_iterator)
                _ = list(_)
    else:
        for i in tqdm(np.random.permutation(index_list), desc='Total progress'):
            
            # cooldown to make sure opencv VideoCapture has been closed
            extractor.extract_face_from_video(i, logger=logger)
        

class TqdmIterator:
    
    def __init__(self, src, position = 0, desc = '', shared_list = None):
        
        self.num = len(src)
        self.current = 0
        
        self.pbar = tqdm(total=self.num, position = position, desc = desc, leave=None)
        
        # self.lock = lock
        
        self.shared_list = shared_list
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        if self.current == self.num:
            raise StopIteration()
            
        ret = self.current
        self.current += 1
        
        with self.shared_list[7]:
            self.pbar.update(1)
        
        return ret
        
class Extractor:
    
    def __init__(self, shared_list):
        
        self.detector_path   = shared_list[0]
        self.sp_path         = shared_list[1]
        self.src_paths  = shared_list[2]
        self.tgt_dir    = shared_list[3]
        self.tgt_fps    = shared_list[4]
        self.tgt_size   = shared_list[5]
        
        self.tmp_dir    = shared_list[6]
        
        self.lock = shared_list[7]
        
        self.available_positions = shared_list[8]
        
        self.shared_list = shared_list

        self.detector = None
        self.sp = None

    def extract_face_from_video(self, i, logger=None):
        src_path = self.src_paths[i]
        separated = str(src_path.absolute()).split(os.sep)
        file_name = src_path.stem
        file_id = separated[-2]
        tgt_path = self.tgt_dir / '{}-{}'.format(file_id, file_name + '.npy')
        logger.debug(f'[{i}] Processing {src_path} -> {tgt_path}')
        if self.detector is None:
            self.detector = dlib.cnn_face_detection_model_v1(self.detector_path)

        if self.sp is None:
            self.sp = dlib.shape_predictor(self.sp_path)

        if not os.path.exists(tgt_path):
            tqdm_position = i
            with self.lock:
                for position_candidate, available in enumerate(self.shared_list[8]):
                    if available:
                        tqdm_position = position_candidate
                        self.shared_list[-1][tqdm_position] = False
                        break

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
            processed_frames = np.zeros((
                num_frames, self.tgt_size[0], self.tgt_size[1], 3), dtype=np.uint8)


            pbar = tqdm(total = num_frames, position=tqdm_position+1,
                            desc='Row {:02d} - index {:02d}: Processing {}'.format(tqdm_position+1, i, src_path.absolute()),
                            leave=None)

            for j in range(num_frames):

                pbar.update(1)

                ret, frame = cap.read()

                if not ret:
                    continue

                # dets = detector(frame, 1)
                dets = safe_detect(logger, self.detector, frame, i)


                num_faces = len(dets)
                if num_faces != 0:

                    faces = dlib.full_object_detections()
                    for detection in dets:
                        detection = detection.rect

                        shape = self.sp(frame, detection)
                        faces.append(shape)

                    chips = dlib.get_face_chips(frame, faces, size=self.tgt_size[0])
                    if chips and isinstance(chips[0], np.ndarray) and chips[0].shape == (
                    self.tgt_size[0], self.tgt_size[1], 3):
                        face_frame = chips[0]
                    else:
                        face_frame = np.zeros(
                            (self.tgt_size[0], self.tgt_size[1], 3), dtype=np.uint8)
                    processed_frames[j] = face_frame
                else:
                    no_face_cnt += 1

            pbar.close()
            cap.release()

            if len(processed_frames) == 0:
                if logger is not None:
                    logger.error(f'No faces detected in {src_path}. {tgt_path} will not be '
                                    'created')
            else:
                processed_frames = np.stack(processed_frames).astype(np.uint8)
                np.save(str(tgt_path), processed_frames)

            gc.collect()

            with self.lock:
                self.shared_list[-1][tqdm_position] = True

        return None

if __name__ == '__main__':
    main()
