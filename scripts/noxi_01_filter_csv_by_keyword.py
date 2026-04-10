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

from pathlib import Path
from tqdm import tqdm
import numpy as np
import csv
import sys
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

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

    src_train_path = tgt_dir / (base_name + 'train.csv')
    src_valid_path = tgt_dir / (base_name + 'valid.csv')
    src_test_path = tgt_dir / (base_name + 'test.csv')

    tgt_train_path = tgt_dir / (base_name + f'train_{keyword.lower()}.csv')
    tgt_valid_path = tgt_dir / (base_name + f'valid_{keyword.lower()}.csv')
    tgt_test_path = tgt_dir / (base_name + f'test_{keyword.lower()}.csv')
    
    # print(tgt_train_path)
    # print(tgt_valid_path)
    # print(tgt_test_path)
    # input('ok?: ')
    
    train_size = 0.8
    valid_size = 0.1
    test_size = 0.1
    
    data = load_csv(src_train_path)
    # pp.pprint(data[:2])
    
    data.extend(load_csv(src_valid_path)[1:])
    data.extend(load_csv(src_test_path)[1:])

    data_columns = data[0]
    data_list = data[1:]
    
    data_list = filter_by_keyword(data_list, keyword)
    
    if len(data_list) == 0:
        logger.error('No keyword matched')
        sys.exit()
    
    train_data, valid_test_data = train_test_split(data_list, train_size=train_size,
                                                   test_size=valid_size + test_size)
    valid_data, test_data = train_test_split(valid_test_data, 
                                             train_size=valid_size / (valid_size + test_size),
                                             test_size=test_size / (valid_size + test_size))
    
    train_data.insert(0, data_columns)
    valid_data.insert(0, data_columns)
    test_data.insert(0, data_columns)
    
    write_csv(tgt_train_path, train_data)
    write_csv(tgt_valid_path, valid_data)
    write_csv(tgt_test_path, test_data)
    

def filter_by_keyword(src, keyword):
    tgt = []
    for row in tqdm(src, desc=f'Filtering by {keyword}'):
        if keyword in row[0] or keyword == "All":
            tgt.append(row)

    return tgt

def load_csv(src_path, delimiter = ','):
    
    with open(src_path, 'r') as f:
        reader = csv.reader(f, delimiter = delimiter)
        src_data = [x for x in reader]
    
    return src_data

def write_csv(tgt_path, tgt_data):
    
    with open(tgt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(tgt_data)

if __name__ == '__main__':
    main()