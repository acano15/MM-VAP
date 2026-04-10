# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parent
sys.path.insert(0, str(base_path.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path = list(dict.fromkeys(sys.path))

import time
import signal
import hydra
from omegaconf import DictConfig
import threading

from src.libs.data.circular_buffer import CCircularBuffer
from src.libs.logger.log import getLogger, load_logger_config
from src.libs.utils.resolvers import OmegaConf

SLEEPING_DELAY = 5

def producer(buffer: CCircularBuffer, count: int = 10, delay: float = 0.3):
    """Simulates a producer filling the buffer."""
    global logger

    for i in range(count):
        success = buffer.add(i, name=f"item_{i}")
        if success:
            logger.info(f"Produced {i}")
        else:
            logger.warning(f"Failed to store item_{i}")
        time.sleep(delay)
    logger.info("Producer finished")


def consumer(buffer: CCircularBuffer, delay: float = 0.5):
    """Simulates a consumer reading from the buffer."""
    global logger

    while not buffer.is_empty():
        item = buffer.get()
        if item is not None:
            logger.info(f"Consumed: {item}")
            time.sleep(delay)
        else:
            time.sleep(0.1)
    logger.info("Consumer finished")

def handle_signal(signum, frame):
    global logger, buffer
    signame = signal.Signals(signum).name
    logger.warning(f"Received {signame}. Stopping buffer")
    buffer.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg_dict: DictConfig) -> None:
    global logger

    OmegaConf.resolve(cfg_dict)

    load_logger_config(cfg_dict.logger)
    logger = getLogger(name="Main")
    logger.info("Starting CircularBuffer test")

    args = sys.argv[1:]
    if len(args) > 0:
        is_streaming = bool(int(args[0]))
    else:
        is_streaming = False

    if len(args) >= 2:
        is_replacing = bool(int(args[1]))
    else:
        is_replacing = False

    buffer = CCircularBuffer(5, is_streaming=is_streaming, is_replacing=is_replacing)

    producer_thread = threading.Thread(target=producer, args=(buffer,))
    time.sleep(2)  # Simulate work
    consumer_thread = threading.Thread(target=consumer, args=(buffer,))

    producer_thread.start()
    consumer_thread.start()

    logger.info("Waiting for threads to finish")
    try:
        logger.debug("Press Ctrl+C to stop.")
        while producer_thread.is_alive() or consumer_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received. Stopping buffer")
        buffer.stop()

    logger.info("Test complete, stopping buffer")
    buffer.stop()

if __name__ == "__main__":
    main()
