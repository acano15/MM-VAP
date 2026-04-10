# -*- coding: utf-8 -*-
import sys
import os
import threading
import pytorch_lightning as pl

from src.libs.logger.log import getLogger


class CManualStopCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self._logger = getLogger(self.__class__.__name__)

        self.stop_training = False
        self.stop_validation = False

        try:
            if os.name == "nt":
                self._logger.info("Using Windows msvcrt for keyboard listener")
                self._start_console_listener()
            else:
                if os.environ.get("DISPLAY") is not None:
                    self._logger.info("Using pynput for keyboard listener")
                    self._start_pynput_listener()
                else:
                    self._logger.info("Using stdin for keyboard listener")
                    self._start_stdin_listener()
        except Exception as e:
            self._logger.warning(f"Error when trying to interact with keyboard: {e}. No manual stopping")

    def _start_console_listener(self):
        import msvcrt
        def listen():
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch == b'\x10':  # Ctrl+P
                        self._logger.warning(
                            "Ctrl+P detected (Windows msvcrt): stopping training after this batch")
                        self.stop_training = True
                        break

        t = threading.Thread(target=listen, daemon=True)
        t.start()
        self.listener_thread = t

    def _start_pynput_listener(self):
        from pynput import keyboard

        def on_press(key):
            # Detect Ctrl+P
            if key == keyboard.KeyCode(char='p') and self.ctrl_pressed:
                self._logger.warning("Ctrl+P detected: stopping training after this batch")
                self.stop_training = True
                return False  # Stop listener

            # Track if Ctrl is pressed
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = True

        def on_release(key):
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.ctrl_pressed = False

        self.ctrl_pressed = False
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

    def _start_stdin_listener(self):
        import termios, tty, select, contextlib, atexit

        def _restore_terminal(fd, old_settings):
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass  # ignore if stdin is already closed

        @contextlib.contextmanager
        def cbreak_mode(fd):
            old_settings = termios.tcgetattr(fd)
            atexit.register(_restore_terminal, fd, old_settings)
            try:
                tty.setcbreak(fd)
                yield
            finally:
                _restore_terminal(fd, old_settings)

        def listen():
            fd = sys.stdin.fileno()
            with cbreak_mode(fd):  # <-- ensures restore on thread exit
                while True:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == '\x10':  # Ctrl+P
                            self._logger.warning("Ctrl+P detected (stdin): stopping training after this batch")
                            self.stop_training = True
                            break

        self.listener_thread = threading.Thread(target=listen, daemon=True)
        self.listener_thread.start()

    def on_train_batch_end(self, trainer, pl_module, batch, *args, **kwargs):
        if self.stop_training:
            self._logger.warning("Stopping training after this batch")
            trainer.should_stop = True
            trainer._terminate_gracefully = True
            if hasattr(self, "listener") and self.listener and getattr(self.listener, "running", False):
                self.listener.stop()

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.stop_training:
            self._logger.warning(
                "Manual stop requested: skipping validation and stopping after this epoch")
            trainer.should_stop = True
            trainer._terminate_gracefully = True
            trainer.limit_val_batches = 0
            try:
                n = len(trainer.val_dataloaders) if trainer.val_dataloaders is not None else 0
                trainer.num_val_batches = [0] * n
                trainer.num_sanity_val_batches = 0
            except Exception:
                pass
            if hasattr(self, "listener") and self.listener and getattr(
                self.listener, "running", False):
                self.listener.stop()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.stop_training:
            trainer.should_stop = True
            trainer._terminate_gracefully = True
            if hasattr(self, "listener") and self.listener and getattr(self.listener, "running", False):
                self.listener.stop()
