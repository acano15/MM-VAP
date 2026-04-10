import logging
from functools import partial, partialmethod
from pathlib import Path
from datetime import datetime
import colorlog
import inspect
from omegaconf import DictConfig, OmegaConf


# --- Custom Log Levels ---
logging.TRACE = 5
logging.addLevelName(logging.TRACE, 'TRACE')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)

logging.DEV = 9
logging.addLevelName(logging.DEV, 'DEV')
logging.Logger.dev = partialmethod(logging.Logger.log, logging.DEV)

logging.RECORD = 25
logging.addLevelName(logging.RECORD, 'RECORD')
logging.Logger.record = partialmethod(logging.Logger.log, logging.RECORD)

LOG_LEVELS = {
    "TRACE": logging.TRACE,
    "DEV": logging.DEV,
    "RECORD": logging.RECORD,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}
STREAM_FORMATTER = ('%(asctime)s [%(threadName)s] [%(name)s] [%(levelname)s] (%(filename)s:%('
                    'lineno)s [%(funcName)s]) - %(message)s')

_logger_instances = {}
_logger_config = None
_logger_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def load_logger_config(config):
    """Load logger configuration from path, dict, or DictConfig."""
    global _logger_config

    if isinstance(config, str):
        cfg = OmegaConf.load(config)
    elif isinstance(config, DictConfig):
        cfg = config
    elif isinstance(config, dict):
        cfg = OmegaConf.create(config)
    else:
        raise TypeError("Logger config must be a file path (str), dict, or DictConfig.")

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    new_config = type("LoggerConfig", (), resolved_cfg)
    if not hasattr(new_config, "format") or not new_config.format:
        new_config.format = STREAM_FORMATTER

    # If we already had a config and loggers exist → refresh them
    if _logger_config is not None and _logger_instances:
        _logger_config = new_config
        for name in list(_logger_instances.keys()):
            # force rebuild by removing, then recreating
            _logger_instances.pop(name)
            _logger_instances[name] = getLogger(name)
    else:
        _logger_config = new_config


def getLogger(name: str):
    global _logger_instances, _logger_config, _logger_timestamp

    if name in _logger_instances:
        return _logger_instances[name]

    if _logger_config is None:
        raise RuntimeError("Logger config not loaded. Call load_logger_config() first.")

    log = logging.getLogger(name)
    log.propagate = False
    log.handlers.clear()

    class_cfg = _logger_config.classes.get(name)
    if class_cfg is None:
        for partial_name in _logger_config.classes:
            if name.startswith(partial_name):
                class_cfg = _logger_config.classes[partial_name]
                break
        else:
            class_cfg = {}

    level_str = class_cfg.get("level", _logger_config.default_level).upper()
    level_num = LOG_LEVELS.get(level_str, logging.WARNING)
    log.setLevel(level_num)

    # --- Console Handler (colorlog) ---
    if class_cfg.get("log_to_console", _logger_config.default_log_to_console):
        color_fmt = "%(log_color)s" + _logger_config.format
        color_formatter = colorlog.ColoredFormatter(
            color_fmt,
            datefmt="%d-%m-%Y %H:%M:%S",
            log_colors={
                'TRACE': 'white',
                'DEV': 'cyan',
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
                'RECORD': 'white',
            }
        )
        sh = colorlog.StreamHandler()
        sh.setFormatter(color_formatter)
        sh.setLevel(level_num)
        sh.addFilter(lambda r: r.levelno != logging.RECORD)
        log.addHandler(sh)

    # --- File Handler ---
    if class_cfg.get("log_to_file", _logger_config.default_log_to_file):
        per_class = class_cfg.get("new_file", _logger_config.default_per_class_file)
        filename = f"{name}.log" if per_class else _logger_config.shared_log_filename
        folder = Path(_logger_config.log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename

        file_formatter = logging.Formatter(_logger_config.format, datefmt="%d-%m-%Y %H:%M:%S")
        fh = logging.FileHandler(path)
        fh.setFormatter(file_formatter)
        file_level_str = class_cfg.get("file_level") or level_str or _logger_config.default_level
        file_level = LOG_LEVELS.get(file_level_str.upper(), level_num)
        fh.setLevel(logging.NOTSET)

        class AlwaysRecordFilter(logging.Filter):
            def filter(self, record):
                return record.levelno >= file_level or record.levelno == logging.RECORD
        fh.addFilter(AlwaysRecordFilter())
        log.addHandler(fh)

    def log_begin(self, method: str = None):
        self.trace(f"[BEGIN] {method}")

    def log_end(self, method: str = None):
        self.trace(f"[END] {method}")

    log.log_begin = log_begin.__get__(log)
    log.log_end = log_end.__get__(log)

    def traced_method(self, func):
        def wrapper(*args, **kwargs):
            is_method = args and hasattr(args[0], "__class__") and hasattr(args[0], "_logger")
            logger = args[0]._logger if is_method else self
            logger.log_begin(func.__name__)
            result = func(*args, **kwargs)
            logger.log_end(func.__name__)
            return result

        return wrapper
    log.traced = traced_method.__get__(log)

    def set_new_name(self, new_name: str):
        """Rename the logger and reapply configuration."""
        _logger_instances.pop(self.name, None)  # Remove old reference
        self.name = new_name
        _logger_instances[new_name] = self
        # Reapply config using the new name
        new_cfg = _logger_config.classes.get(new_name, {})
        fallback_cfg = _logger_config.classes.get(
            next((k for k in _logger_config.classes if new_name.startswith(k)), None), {}
            )
        effective_cfg = {**fallback_cfg, **new_cfg}

        level_str = effective_cfg.get("level", _logger_config.default_level).upper()
        level_num = LOG_LEVELS.get(level_str, logging.WARNING)
        for handler in self.handlers:
            handler.setLevel(level_num)

    log.set_new_name = set_new_name.__get__(log)

    _logger_instances[name] = log
    return log


def set_logger_level(name: str, level: str):
    """Update the log level for an existing logger instance and its console handler."""
    if name not in _logger_instances:
        raise ValueError(f"Logger '{name}' is not initialized.")

    logger = _logger_instances[name]
    level = level.upper()
    level_num = LOG_LEVELS.get(level)

    if level_num is None:
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(logging.TRACE)

    # Update level for console handler (StreamHandler only)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level_num)

    logger.trace(f"Console log level changed to {level} for logger '{name}'")
