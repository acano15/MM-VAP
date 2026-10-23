# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 17/01/2023
# version = '0.1'
# ---------------------------------------------------------------------------
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import argparse
import time
import json

from src.libs.audio_io.audio_reader import CAudioReader
from src.libs.data.audio_data.audio_format import CAudioFormat
from hydra import compose, initialize
from src.libs.logger.log import getLogger, load_logger_config
from .test_audio_io_listener import CAudioIOListenerTest
from src.libs.thread.signal_handler import get_signal_handler

# DEFINITIONS
SLEEP_TIME_IN_SECONDS = 0.5
interrupted = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


config_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(config_path, "config", "", "config.json")) as json_data_file:
    config_data = json.load(json_data_file)["test"]

def _signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("Received interrupt signal. Shutting down...")
    os._exit(1)


signal_handler = get_signal_handler()
signal_handler.register(_signal_handler)


def main():
    try:
        global interrupted

        result = 0
        my_parser = argparse.ArgumentParser(description='CAMERA TEST')
        my_parser.add_argument('-log_level', '--log_level',
                               type=str, default="WARNING", nargs='?',
                               help='Logging level (DEBUG, INFO, WARNING, ERROR, DEV, TRACE)',
                               choices=[level.upper() for level in ['DEBUG', 'INFO', 'WARNING',
                                        'ERROR', 'DEV', 'TRACE']])
        # Mic/Audio arguments
        my_parser.add_argument('-source', '--source', type=str, default=0, nargs='?',
                               help='Name of the audio (or num mic) with the file format')
        my_parser.add_argument('-threaded', '--threaded', type=str2bool, default=False,
                               nargs='?', help='Display audio signal in True mode')

        # Display arguments
        my_parser.add_argument('-display_signal', '--display_signal', type=str2bool,
                               default=True, nargs='?', help='Display audio signal in True mode')

        args = my_parser.parse_args()
        log_level = args.log_level.upper()
        load_logger_config({"log_dir": "logs", "shared_log_filename": "test.log",
                            "default_level": log_level, "default_log_to_file": False,
                            "default_log_to_console": True,
                            "default_per_class_file": False, "classes": {}})
        logger = getLogger("AudioIO_test")
        try:
            source = int(args.source)
        except BaseException:
            source = args.source
        threaded = args.threaded
        display_signal = args.display_signal

        with initialize(config_path="../../../configs", version_base=None):
            cfg_dict = compose(config_name="config")
        audio_io_config = cfg_dict.io.audio_io
        chunk_size = audio_io_config.chunk_size
        num_channels = audio_io_config.audio_format.num_channels
        sample_rate = audio_io_config.audio_format.sample_rate
        format = audio_io_config.audio_format.format
        audio_format = CAudioFormat(sample_rate, num_channels, format)
        audio_reader_manager = CAudioReader(
                source=source,
                threaded=threaded,
                chunk_size=chunk_size,
                format=audio_format,
                show_signal=display_signal,
                resample_per=100,
                save_audio=config_data["save_audio"],
                save_interval=config_data["save_interval"],
                audio_path=config_data["audio_path"])

        test_listener = CAudioIOListenerTest(log_level)
        audio_reader_manager.add_listener(test_listener)
        audio_reader_manager.start()
        while audio_reader_manager.is_active() and not interrupted:
            logger.debug("AudioIO test: waiting for the pending chunks")
            time.sleep(SLEEP_TIME_IN_SECONDS)
    except Exception as e:
        result = 1
        print(e)


if __name__ == "__main__":
    sys.exit(main())
