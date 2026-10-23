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
import cv2
import json

from src.libs.image_io.source_reader import CCaptureReader
from src.libs.data.image_data.image_format import CImageFormat
from src.libs.logger.log import getLogger, load_logger_config
from src.libs.thread.signal_handler import get_signal_handler
from .test_image_io_listener import CImageIOListenerTest

# DEFINITIONS
SLEEP_TIME_IN_SECONDS = 0.5
interrupted = False

config_path = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(config_path, "config", "", "config.json")) as json_data_file:
    config_data = json.load(json_data_file)["test"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
        # Cam/Video arguments
        my_parser.add_argument('-use_cam', '--use_cam', type=str2bool, default=True, nargs='?',
                               help='Turn on the camera in True mode')
        my_parser.add_argument('-source', '--source', type=str, default=0, nargs='?',
                               help='Name of the video (or num cam) with the file format')
        my_parser.add_argument('-threaded', '--threaded', type=str2bool, default=True, nargs='?',
                               help='Turn on the camera in thread mode')
        my_parser.add_argument('-resize_per', '--resize_per', type=int, default=100, nargs='?',
                               help='Resize proportion to the original image')
        my_parser.add_argument('-save_interval', '--save_interval', type=int, default=60, nargs='?',
                               help='Interval in seconds to save the video')
        my_parser.add_argument('-fps_per', '--fps_per', type=int, default=100, nargs='?',
                               help='FPS percentage to modify')
        my_parser.add_argument('-save_video', '--save_video', type=str2bool, default=False, nargs='?',
                               help='Switch the video saver in True mode. Save video' +
                                    'in the same path but with _result in the name')
        my_parser.add_argument('-name_save', '--name_save', type=str, default="", nargs='?',
                               help='Name of the video to save')
        my_parser.add_argument('-save_path', '--save_path', type=str, default="", nargs='?',
                               help='Path save the video')

        # Display arguments
        my_parser.add_argument('-display_video', '--display_video', type=str2bool, default=True, nargs='?',
                               help='Display video in True mode')
        my_parser.add_argument('-display_fps', '--display_fps', type=str2bool, default=False, nargs='?',
                               help='Display fps in True mode')
        my_parser.add_argument('-display_date', '--display_date', type=str2bool, default=False, nargs='?',
                               help='Display date in True mode')

        args = my_parser.parse_args()
        log_level = args.log_level.upper()
        load_logger_config({"log_dir": "logs", "shared_log_filename": "test.log",
                            "default_level": log_level, "default_log_to_file": False,
                            "default_log_to_console": True,
                            "default_per_class_file": False, "classes": {}})
        logger = getLogger("VideoIO_test")

        use_cam = args.use_cam
        try:
            source = int(args.source)
        except BaseException:
            source = args.source
        threaded = args.threaded
        resize_per = args.resize_per
        save_interval = args.save_interval
        fps_per = args.fps_per

        save_video = args.save_video
        name_save = args.name_save
        save_path = args.save_path

        display_video = args.display_video
        display_fps = args.display_fps
        display_date = args.display_date

        fps = config_data["image_io"]["image_format"]["fps"]
        width = config_data["image_io"]["image_format"]["width"]
        height = config_data["image_io"]["image_format"]["height"]
        num_channels = config_data["image_io"]["image_format"]["num_channels"]

        image_format = CImageFormat(fps=fps, width=width, height=height,
                                    number_channels=num_channels)
        if not save_path:
            save_path = config_data["video_path"]
        if name_save:
            save_path = os.path.join(save_path, name_save)

        video_reader_manager = CCaptureReader(source=source, format=None,
                                              threaded=threaded, show_video=display_video,
                                              resize_per=resize_per, save_video=save_video,
                                              save_interval=save_interval, fps_per=fps_per,
                                              write_fps=display_fps, write_date=display_date,
                                              video_path=save_path)
        test_listener = CImageIOListenerTest(log_level)
        video_reader_manager.add_listener(test_listener)
        video_reader_manager.start()
        try:
            while video_reader_manager.is_active() and not interrupted:
                logger.debug("VideoIO test: waiting for the pending frames")
                time.sleep(SLEEP_TIME_IN_SECONDS)
        except Exception as e:
            logger.error(e)
            raise e
    except Exception as e:
        result = 1
        print(e)


if __name__ == "__main__":
    sys.exit(main())
