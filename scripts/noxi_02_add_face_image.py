# -*- coding: utf-8 -*-
"""

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

import os.path
from pathlib import Path
import csv
import numpy as np
import hydra
from omegaconf import DictConfig

from log import load_logger_config, getLogger
from utils import OmegaConf


@hydra.main(config_path="../src/configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig):
    load_logger_config(cfg_dict.logger)
    logger = getLogger("Main")

    OmegaConf.resolve(cfg_dict)

    src_dir = Path(cfg_dict.dataset.path)
    tgt_dir = Path(src_dir.parent / "noxi_extracted")

    keyword = cfg_dict.dataset.dataset_selected.keyword
    chunk_duration = cfg_dict.data.audio_duration
    horizon = cfg_dict.data.vad_horizon
    stride = cfg_dict.data.audio_overlap
    base_name = f"audio_duration_{chunk_duration}_horizon_{horizon}_stride_{stride}_"

    src_train_path = tgt_dir / (base_name + f'train_{keyword.lower()}.csv')
    src_valid_path = tgt_dir / (base_name + f'valid_{keyword.lower()}.csv')
    src_test_path = tgt_dir / (base_name + f'test_{keyword.lower()}.csv')
    extension = ".csv"
    new_train_path_name = src_train_path.with_name(
        src_train_path.name.replace(extension, f"_faces{extension}"))
    new_valid_path_name = src_valid_path.with_name(
        src_valid_path.name.replace(extension, f"_faces{extension}"))
    new_test_path_name = src_test_path.with_name(
        src_test_path.name.replace(extension, f"_faces{extension}"))
    src_tgt_pairs = [(src_train_path, new_train_path_name),
                     (src_valid_path, new_valid_path_name),
                     (src_test_path, new_test_path_name)]

    # WRITE = False
    WRITE = True
    
    for src_path, tgt_path in src_tgt_pairs:
        src_data = load_csv(src_path)
        
        src_columns = src_data[0]
        src_list = src_data[1:]
        
        # for i in range(len(src_columns)):
        #     print('{:2d}, {}'.format(i, src_columns[i]))
        # pp.pprint(src_list[0])
        
        tgt_list = []
        tgt_columns = src_columns.copy()
        tgt_columns.insert(8, 'face_im_path1')
        tgt_columns.insert(13, 'face_im_path2')
        
        # for i in range(len(tgt_columns)):
        #     print('{:2d}, {}'.format(i, tgt_columns[i]))
            
        for row in src_list:
            add_row = True
            tmp_src_path = Path(row[4]).parent
            novice_name = f"{row[0]}-novice.video.npy"
            tmp_tgt_path = tmp_src_path / novice_name
            if os.path.exists(tmp_tgt_path):
                try:
                    _ = np.load(tmp_tgt_path, allow_pickle=False, mmap_mode='r')
                    row.insert(8, tmp_tgt_path)
                except Exception as e:
                    add_row = False
                    print(f"Failed to load {tmp_tgt_path}: {e}")
            else:
                add_row = False
                print(f"File {tmp_tgt_path} doesn't exist. Skipped")
            # print(tmp_src_path)
            # print(tmp_tgt_path)

            tmp_src_path = Path(row[8]).parent
            expert_name = f"{row[0]}-expert.video.npy"
            tmp_tgt_path = tmp_src_path / expert_name
            if os.path.exists(tmp_tgt_path):
                try:
                    _ = np.load(tmp_tgt_path, allow_pickle=False, mmap_mode='r')
                    row.insert(13, tmp_tgt_path)
                except Exception as e:
                    add_row = False
                    print(f"Failed to load {tmp_tgt_path}: {e}")
            else:
                add_row = False
                print(f"File {tmp_tgt_path} doesn't exist. Skipped")

            # print(tmp_src_path)
            # print(tmp_tgt_path)
            
            if add_row:
                tgt_list.append(row)

            # input()
            
        tgt_list.insert(0, tgt_columns)
        
        # pp.pprint(tgt_list[0])
        # print(np.shape(tgt_list))
        
        if WRITE:
            write_csv(tgt_path, tgt_list)


def load_csv(src_path, delimiter = ','):
    
    with open(src_path, 'r') as f:
        reader = csv.reader(f, delimiter = delimiter)
        src_data = [x for x in reader]
    
    return src_data

def write_csv(tgt_path, tgt_data):
    
    with open(tgt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tgt_data)
    
    print('complete to write:', tgt_path)

if __name__ == '__main__':
    main()
    
"""
[
 [
  [3.77, 4.51], [5.25, 7.84], [8.46, 14.1], [16.73, 17.39], [17.72, 20.33]
 ], 
 [
  [13.16, 13.4], [13.98, 14.13], [14.56, 17.74], [20.73, 22.0]                                                                              
 ]
]
"""
