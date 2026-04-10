# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_path))
sys.path = list(dict.fromkeys(sys.path))

import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv
import hydra
from omegaconf import DictConfig
from multiprocessing import Pool, Manager, Lock
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import torchaudio
import shutil
import random

from VAP.scripts.face_extractor import CFaceExtractor
from src.libs.face_detector import CFaceDetector
from src.libs.face_detector import CLandmarksDetector
from src.libs.logger.log import load_logger_config, getLogger
from src.libs.utils import repo_root, select_platform_path, OmegaConf


def get_partial_vad(src_list, start_time, end_time):
    
    pass

    tgt_list = [[], []]
    
    for ch in range(2):
        for row in src_list[ch]:
            
            if ((start_time < row[0]) and (row[0] < end_time)) or ((start_time < row[1]) and (row[1] < end_time)):
                if (row[0] < start_time):
                    tmp_start_time = start_time
                elif (start_time <= row[0]):
                    tmp_start_time = row[0]
                
                if (row[1] < end_time):
                    tmp_end_time = row[1]
                elif (end_time <= row[1]):
                    tmp_end_time = end_time
                
                tgt_list[ch].append([tmp_start_time, tmp_end_time])

    return tgt_list


def get_vad_data(src_path):
    vad_list = load_vad_csv(src_path, ' ')
    return 0, 0, vad_list


def load_vad_csv(src_path, delimiter = ','):
    src_data = load_csv(src_path, delimiter)
    
    tgt_data = [[], []]
    for row in src_data:
        if row[-1] == '[speech]':
           if 'novice' in str(src_path.absolute()):
               tgt_data[0].append([float(row[1]), float(row[2])])
           else:
               tgt_data[1].append([float(row[1]), float(row[2])])

    return tgt_data


def load_csv(src_path, delimiter=','):
    with open(src_path, 'r') as f:
        reader = csv.reader(f, delimiter = delimiter)
        src_data = [x for x in reader]
    
    return src_data


def write_csv(tgt_path, tgt_data):
    with open(tgt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tgt_data)


def load_nonverbal_data(src_path):
    src_data = load_csv(src_path)

    gaze_data = [['gaze_x', 'gaze_y']]
    face_data = [['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']]
    head_data = [['head_x', 'head_y', 'head_z']]
    body_data = [['pose_1_x', 'pose_1_y', 'pose_1_confidence', 'pose_2_x', 'pose_2_y', 'pose_2_confidence', 'pose_3_x', 'pose_3_y', 'pose_3_confidence', 'pose_4_x', 'pose_4_y', 'pose_4_confidence', 'pose_5_x', 'pose_5_y', 'pose_5_confidence', 'pose_6_x', 'pose_6_y', 'pose_6_confidence', 'pose_7_x', 'pose_7_y', 'pose_7_confidence']]

    for row in src_data[1:]:
        gaze_data.append(row[1:4])
        face_data.append(row[4:21])
        head_data.append(row[21:24])
        body_data.append(row[24:])

    return face_data, gaze_data, head_data, body_data


def filter_by_keyword(src, keyword):
    tgt = []
    for row in tqdm(src, desc=f'Filtering by {keyword}'):
        if keyword in row[0] or keyword == "All":
            tgt.append(row)

    return tgt

class TqdmIterator:

    def __init__(self, src, position=0, desc='', lock=None):
        self.num = len(src)
        self.current = 0
        self.pbar = tqdm(total=self.num, position=position, desc=desc, leave=None)
        self.lock = lock

    def __iter__(self):

        return self

    def __next__(self):

        if self.current == self.num:
            raise StopIteration()

        ret = self.current
        self.current += 1

        with self.lock:
            self.pbar.update(1)

        return ret


@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig):
    load_logger_config(cfg_dict.logger)
    logger = getLogger("Main")

    OmegaConf.resolve(cfg_dict)

    src_dir = Path(cfg_dict.dataset.path)
    tgt_dir = Path(src_dir.parent / "noxi_extracted")
    os.makedirs(tgt_dir, exist_ok=True)

    chunk_duration = cfg_dict.data.audio_duration
    horizon = cfg_dict.data.vad_horizon
    stride = cfg_dict.data.audio_overlap
    base_name = f"audio_duration_{chunk_duration}_horizon_{horizon}_stride_{stride}_"

    session_folder = (src_dir / "sessions").resolve()
    files = [x for x in session_folder.glob('**/*') if x.is_file() and not x.parent.name[
        0].isdigit()]
    files = [x for x in files if '.gitkeep' not in str(x)]

    use_clean_audio = cfg_dict.data.audio.use_clean
    if use_clean_audio:
        files = [x for x in files if
                 x.suffix != '.wav' or x.stem.endswith('_clean') or not x.with_name(
                     f'{x.stem}_clean{x.suffix}').exists()]
    else:
        files = [x for x in files if x.suffix != '.wav' or not x.stem.endswith('_clean')]

    use_computed_vad = cfg_dict.data.use_computed_vad
    if use_computed_vad:
        files = [x for x in files if
                 x.suffix != '.txt' or x.stem.endswith('_computed') or not x.with_name(
                     f'{x.stem}_computed{x.suffix}').exists()]
    else:
        files = [x for x in files if x.suffix != '.txt' or not x.stem.endswith('_computed')]

    logger.info(f'Found {len(files)} files in {session_folder}')

    data_list = [['id', 'start', 'end', 'audio_path', 'face_path1', 'gaze_path1', 'head_path1',
                  'body_path1', 'face_path2', 'gaze_path2', 'head_path2', 'body_path2', 'vad_list',
                  'session', 'dataset']]

    # 1. Prepare data from files
    data_dict = {}
    for file in tqdm(files, desc='Collecting data src'):
        logger.debug(f'Processing file: {file}')
        separated = str(file.absolute()).split(os.sep)
        file_name = separated[-1]
        file_id = separated[-2]

        # input()
        if not (file_id in data_dict.keys()):

            data_dict[file_id] = {}
            data_dict[file_id]['session'] = 0
            data_dict[file_id]['dataset'] = 'noxi'

        if 'audio' in file_name and file_name.endswith(".wav"):
            if 'novice' in file_name:
                tgt_path = tgt_dir / '{}-{}'.format(file_id, file_name)
                logger.dev(f'Processing novice audio file: {tgt_path}')
                data_dict[file_id]['audio_path1'] = str(tgt_path.absolute())
                if not tgt_path.exists():
                    shutil.copy(file.absolute(), tgt_path.absolute())

            if 'expert' in file_name:
                tgt_path = tgt_dir / '{}-{}'.format(file_id, file_name)
                logger.dev(f'Processing expert audio file: {tgt_path}')
                data_dict[file_id]['audio_path2'] = str(tgt_path.absolute())
                if not tgt_path.exists():
                    shutil.copy(file.absolute(), tgt_path.absolute())

            if 'mix' in file_name:
                logger.dev(f'Skipping mix audio file: {file_name}. Already processed')
                continue

            tmp_file_name = file_name
            tmp_file_name = tmp_file_name.replace('novice', 'mix')
            tmp_file_name = tmp_file_name.replace('expert', 'mix')
            tgt_path = tgt_dir / '{}-{}'.format(file_id, tmp_file_name)

            if (('audio_path1' in data_dict[file_id].keys()) and
                ('audio_path2' in data_dict[file_id].keys())):
                data_dict[file_id]['audio_path'] = str(tgt_path.absolute())
                if not tgt_path.exists():
                    wav1 = AudioSegment.from_wav(data_dict[file_id]['audio_path1'])
                    wav2 = AudioSegment.from_wav(data_dict[file_id]['audio_path2'])
                    wav1 = wav1[:min(len(wav1), len(wav2))]
                    wav2 = wav2[:min(len(wav1), len(wav2))]
                    wav_mix = AudioSegment.from_mono_audiosegments(wav1, wav2)
                    logger.dev(f'Generating mix audio file: {tgt_path}')
                    wav_mix.export(tgt_path, format='wav')
        elif 'non_varbal' in file_name:
            face_data, gaze_data, head_data, body_data = load_nonverbal_data(file.absolute())
            if file_name.startswith("non_varbal_"):
                if 'novice' in file_name:
                    tgt_path = tgt_dir / '{}-face_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing novice non-verbal face file: {tgt_path}')
                    data_dict[file_id]['face_path1'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, face_data)

                    tgt_path = tgt_dir / '{}-gaze_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing novice non-verbal gaze file: {tgt_path}')
                    data_dict[file_id]['gaze_path1'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, gaze_data)

                    tgt_path = tgt_dir / '{}-head_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing novice non-verbal head file: {tgt_path}')
                    data_dict[file_id]['head_path1'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, head_data)

                    tgt_path = tgt_dir / '{}-body_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing novice non-verbal body file: {tgt_path}')
                    data_dict[file_id]['body_path1'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, body_data)
                else:
                    tgt_path = tgt_dir / '{}-face_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing expert non-verbal face file: {tgt_path}')
                    data_dict[file_id]['face_path2'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, face_data)

                    tgt_path = tgt_dir / '{}-gaze_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing expert non-verbal gaze file: {tgt_path}')
                    data_dict[file_id]['gaze_path2'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, gaze_data)

                    tgt_path = tgt_dir / '{}-head_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing expert non-verbal head file: {tgt_path}')
                    data_dict[file_id]['head_path2'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, head_data)

                    tgt_path = tgt_dir / '{}-body_{}'.format(file_id, file_name.split('_')[-1])
                    logger.dev(f'Processing expert non-verbal body file: {tgt_path}')
                    data_dict[file_id]['body_path2'] = str(tgt_path.absolute())
                    if not tgt_path.exists():
                        write_csv(tgt_path, body_data)
        elif file_name.startswith("vad_") and file_name.endswith(".txt"):
            start_int, end_int, vad_list = get_vad_data(file)
            data_dict[file_id]['start'] = start_int
            data_dict[file_id]['end'] = end_int

            if 'vad_list' in data_dict[file_id].keys():
                data_dict[file_id]['vad_list'][0].extend(vad_list[0])
                data_dict[file_id]['vad_list'][1].extend(vad_list[1])
                logger.dev(f'Appending vad_list for file {file_id}')
            else:
                data_dict[file_id]['vad_list'] = vad_list

    for file_id in tqdm(data_dict.keys(), desc='Creating data chunks'):
        if "audio_path" not in data_dict[file_id]:
            logger.warning(f"Discarding file {file_id}. 'audio_path' not found")
            continue

        x, sr = torchaudio.load(data_dict[file_id]['audio_path'])
        audio_duration = x.shape[1] / sr
        tmp_time = 0
        while (tmp_time + chunk_duration + horizon) < audio_duration:
            try:
                logger.trace(
                    '{:.2f} to {:.2f} (audio_duration: {:.2f})'.format(
                        tmp_time, tmp_time + chunk_duration, audio_duration))

                tmp_data_dict = data_dict[file_id].copy()
                audio_path = tmp_data_dict.get('audio_path')
                vad_list = tmp_data_dict.get('vad_list')

                start_time = tmp_time
                end_time = tmp_time + chunk_duration

                tmp_vad_list = get_partial_vad(vad_list, start_time, end_time + horizon)

                # Optional non-verbal block
                face_path1 = tmp_data_dict.get('face_path1')
                gaze_path1 = tmp_data_dict.get('gaze_path1')
                head_path1 = tmp_data_dict.get('head_path1')
                body_path1 = tmp_data_dict.get('body_path1')
                face_path2 = tmp_data_dict.get('face_path2')
                gaze_path2 = tmp_data_dict.get('gaze_path2')
                head_path2 = tmp_data_dict.get('head_path2')
                body_path2 = tmp_data_dict.get('body_path2')

                data_list.append(
                    [file_id,  # 0
                     start_time,
                     end_time,
                     audio_path,
                     face_path1,
                     gaze_path1,  # 5
                     head_path1,
                     body_path1,
                     face_path2,
                     gaze_path2,
                     head_path2,  # 10
                     body_path2,
                     tmp_vad_list,  # 12
                     tmp_data_dict['session'],
                     tmp_data_dict['dataset']])

                tmp_time += stride
            except KeyError as e:
                logger.error(f"KeyError: {e} in file {file_id}. Skipping this file")
                break

    data_columns = data_list[0]
    data_path = tgt_dir / (base_name + 'data.csv')
    write_csv(data_path, data_list)

    logger.info(f'Data saved to {data_path}. Total size : {len(data_list) - 1} rows')

    # 2. Filter csv by keyword
    keyword = cfg_dict.dataset.dataset_selected.keyword
    data_list_keyword = filter_by_keyword(data_list, keyword)
    data_list_keyword.insert(0, data_columns)
    data_path_keyword = tgt_dir / (base_name + f'data_{keyword.lower()}.csv')
    write_csv(data_path_keyword, data_list_keyword)

    # 3. Extract faces from data
    files_to_extract = [x for x in session_folder.glob('**/*.mp4') if x.is_file() and
                        not x.parent.name[0].isdigit() and
                        keyword.lower() in str(x.parent).lower() if keyword != "All"]

    tgt_size = (cfg_dict.general.width, cfg_dict.general.height)
    tgt_fps = None
    tmp_dir = None

    use_multiprocess = False
    num_process = 3
    if tgt_fps != None:
        tmp_dir = Path("{}_{}".format(str(src_dir), tgt_fps))
        if tmp_dir.exists():
            logger.error(f'{str(tmp_dir)} exists')
            sys.exit()
        else:
            tmp_dir.mkdir()

    manager = Manager()
    lock = manager.Lock()
    available_positions = manager.list([True for _ in range(num_process)])

    face_detector = CFaceDetector(cfg_dict.recognition.face_detector)
    landmarks_detector = CLandmarksDetector(cfg_dict.recognition.landmarks_detector)
    extractor = CFaceExtractor(face_detector, landmarks_detector, files_to_extract, tgt_dir, tgt_fps,
                               tgt_size, tmp_dir, lock, available_positions)

    index_list = list(range(len(files_to_extract)))
    if use_multiprocess:
        tqdm_iterator = TqdmIterator(
            index_list, position=0,
            desc='{:02d}: Total progress'.format(0), lock=lock)

        logger.debug(f'use multiprocessing with {num_process} processes')
        with Pool(processes=num_process) as p:
            _ = p.imap(extractor.extract_face_from_video, tqdm_iterator)
            _ = list(_)
    else:
        for i in tqdm(np.random.permutation(index_list), desc='Total progress'):
            # cooldown to make sure opencv VideoCapture has been closed
            extractor.extract_face_from_video(i)

    # 4. Split data into train, valid, test
    train_size = 0.7
    valid_size = 0.15
    test_size = 0.15

    # group by speaker/session id
    file_ids = list({row[0] for row in data_list_keyword[1:]})
    if cfg_dict.train.training_features.shuffle:
        random.seed(cfg_dict.train.training_features.seed)
        random.shuffle(file_ids)

    n_total = len(file_ids)
    n_train = int(n_total * train_size)
    n_valid = int(n_total * valid_size)
    if n_valid == 0:
        n_valid = 1

    n_test = n_total - n_train - n_valid
    assert n_train > 0, "Train samples must be greater than 0"
    assert n_valid > 0, "Validation samples must be greater than 0"
    assert n_test > 0, "Test samples must be greater than 0"
    split1 = n_train
    split2 = n_train + n_valid
    train_ids = set(file_ids[:split1])
    logger.debug(f"train ids: {train_ids}")
    valid_ids = set(file_ids[split1:split2])
    logger.debug(f"valid ids: {valid_ids}")
    test_ids = set(file_ids[split2:])
    logger.debug(f"test ids: {test_ids}")
    assert train_ids.isdisjoint(valid_ids), "Overlap found between train and valid IDs"
    assert train_ids.isdisjoint(test_ids), "Overlap found between train and test IDs"
    assert valid_ids.isdisjoint(test_ids), "Overlap found between valid and test IDs"

    train_data = [row for row in data_list_keyword[1:] if row[0] in train_ids]
    valid_data = [row for row in data_list_keyword[1:] if row[0] in valid_ids]
    test_data = [row for row in data_list_keyword[1:] if row[0] in test_ids]

    tgt_train_path = tgt_dir / (base_name + f'train_{keyword.lower()}.csv')
    logger.info(f'Train data saved to {tgt_train_path}. Size: {len(train_data) - 1} rows')
    train_data.insert(0, data_columns)
    write_csv(tgt_train_path, train_data)

    tgt_valid_path = tgt_dir / (base_name + f'valid_{keyword.lower()}.csv')
    logger.info(f'Valid data saved to {tgt_valid_path}. Size: {len(valid_data) - 1} rows')
    valid_data.insert(0, data_columns)
    write_csv(tgt_valid_path, valid_data)

    tgt_test_path = tgt_dir / (base_name + f'test_{keyword.lower()}.csv')
    logger.info(f'Test data saved to {tgt_test_path}. Size: {len(test_data) - 1} rows')
    test_data.insert(0, data_columns)
    write_csv(tgt_test_path, test_data)

    logger.info(f'total_size {len(data_list_keyword) - 1}')
    logger.info(
        'train_size: {:} ({:.2f})'.format(
            len(train_data) - 1, (len(train_data) - 1) / (len(data_list_keyword) - 1)))
    logger.info(
        'valid_size: {:} ({:.2f})'.format(
            len(valid_data) - 1, (len(valid_data) - 1) / (len(data_list_keyword) - 1)))
    logger.info(
        'test_size: {:} ({:.2f})'.format(
            len(test_data) - 1, (len(test_data) - 1) / (len(data_list_keyword) - 1)))

    # 5. Add faces to data
    for src_path in [tgt_train_path, tgt_valid_path, tgt_test_path]:
        tgt_path = src_path
        logger.info(f'Processing file {src_path} to add face images')
        src_data = load_csv(src_path)

        src_columns = src_data[0]
        src_list = src_data[1:]

        tgt_list = []
        tgt_columns = src_columns.copy()
        tgt_columns.insert(8, 'face_im_path1')
        tgt_columns.insert(13, 'face_im_path2')

        cache_face_checked = {}
        for row in tqdm(src_list, desc=f'Processing rows from file {src_path}'):
            add_row = True
            novice_name = f"{row[0]}-novice.video.npy"
            tmp_tgt_path = tgt_dir / novice_name
            if os.path.exists(tmp_tgt_path):
                try:
                    if tmp_tgt_path not in cache_face_checked.keys():
                        logger.dev(f'Loading novice face from {tmp_tgt_path}. Checking if it is valid')
                        _ = np.load(tmp_tgt_path, allow_pickle=False, mmap_mode='r')
                        cache_face_checked[tmp_tgt_path] = True

                    row.insert(8, tmp_tgt_path)
                except Exception as e:
                    add_row = False
                    logger.error(f"Failed to load {tmp_tgt_path}: {e}")
            else:
                add_row = False
                logger.warning(f"File {tmp_tgt_path} doesn't exist. Skipped")

            expert_name = f"{row[0]}-expert.video.npy"
            tmp_tgt_path = tgt_dir / expert_name
            if os.path.exists(tmp_tgt_path):
                try:
                    logger.dev(f'Loading expert face from {tmp_tgt_path}. Checking if it is valid')
                    _ = np.load(tmp_tgt_path, allow_pickle=False, mmap_mode='r')
                    row.insert(13, tmp_tgt_path)
                except Exception as e:
                    add_row = False
                    logger.error(f"Failed to load {tmp_tgt_path}: {e}")
            else:
                add_row = False
                logger.warning(f"File {tmp_tgt_path} doesn't exist. Skipped")

            if add_row:
                tgt_list.append(row)

        tgt_list.insert(0, tgt_columns)
        logger.info(f'Writing data to {tgt_path}. Size: {len(tgt_list) - 1} rows')
        write_csv(tgt_path, tgt_list)


if __name__ == '__main__':
    main()
