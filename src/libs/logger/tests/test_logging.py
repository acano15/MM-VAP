# coding: UTF-8
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_path.parent))
sys.path = list(dict.fromkeys(sys.path))

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf

from logger.log import load_logger_config, getLogger, set_logger_level


def log_all_levels(logger, label=""):
    logger.trace(f'{label} Message TRACE')
    logger.dev(f'{label} Message DEV')
    logger.debug(f'{label} Message DEBUG')
    logger.info(f'{label} Message INFO')
    logger.warning(f'{label} Message WARNING')
    logger.error(f'{label} Message ERROR')
    logger.critical(f'{label} Message CRITICAL')
    logger.record(f'{label} Message RECORD')
    print("")


parser = argparse.ArgumentParser(description='LOGGER TEST')
parser.add_argument('-log_level', '--log_level', type=str, default="",
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'DEV', 'TRACE'],
                    help='Logging level override (optional)')
parser.add_argument('-config_file', '--config_file', type=str, default="",
                    help='Path to YAML logger configuration file')
args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig) -> None:
    log_config_file = cfg_dict.logger
    if len(args.config_file) != 0:
        log_config_file = args.config_file
        print(f"Loading file from {log_config_file}")

    load_logger_config(log_config_file)

    logger_main = getLogger("Main")
    if len(args.log_level) != 0:
        set_logger_level("Main", args.log_level)

    logger_main.info("Logger test started")
    log_all_levels(logger_main, "Main")

    logger1 = getLogger("Test_logger1")
    log_all_levels(logger1, "Test_logger1")

    @logger1.traced
    def example_traced_1():
        logger1.info("Inside traced function 1")
        print("")
    example_traced_1()

    logger2 = getLogger("Test_logger2")
    log_all_levels(logger2, "Test_logger2")

    logger3 = getLogger("Test_logger3")
    log_all_levels(logger3, "Test_logger3")
    logger3.trace("Test_logger3 showing trace messages")
    logger3.log_begin()
    logger3.log_end()

    logger4 = getLogger("Test_logger4")
    log_all_levels(logger4, "Test_logger4")
    @logger4.traced
    def example_traced_2():
        logger4.info("Inside traced function 2")
        print("")
    example_traced_2()

    logger5 = getLogger("Test_logger5_partial")
    log_all_levels(logger5, "Test_logger5_partial")
    logger5.set_new_name("logger6")
    log_all_levels(logger5, "logger6")


if __name__ == "__main__":
    main()
