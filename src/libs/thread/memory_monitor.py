# -*- coding: utf-8 -*-
"""
#----------------------------------------------------------------------------
# Created By  : Cano M. Antonio @acano15
# Created Date: 07/03/2023
# version = '0.4' (with tracemalloc and memory leak detection enhancements)
# ---------------------------------------------------------------------------
"""

import os
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path = list(dict.fromkeys(sys.path))


import threading
import psutil
import time
import tracemalloc
import objgraph
import platform
import subprocess
import traceback
import weakref
from src.libs.thread.signal_handler import get_signal_handler  # Custom signal handler import


class CMemoryMonitor(threading.Thread):
    """
    A class to monitor memory usage, detect variable names holding large objects,
    and trace the origin of memory leaks, running in a separate CMD or terminal window.
    """

    def __init__(self, interval: int = 5, threshold: int = 1024 ** 2):
        """
        Initializes the memory monitor with the specified interval and threshold.

        Args:
            interval (int): The time interval in seconds between memory checks.
            threshold (int): Minimum object size (in bytes) to be considered 'large'.
        """
        threading.Thread.__init__(self)
        self._interval = interval
        self._threshold = threshold
        self._stop_event = threading.Event()

        # Registering custom signal handler for clean shutdown
        signal_handler = get_signal_handler()
        signal_handler.register(self._signal_handler)

    def start_in_new_cmd(self):
        """
        Starts the memory monitor by opening a new terminal window (CMD on Windows, terminal on Linux)
        and running the monitoring process in that terminal.
        """
        script_path = os.path.abspath(__file__)

        # Open a new terminal or CMD window
        if platform.system() == "Windows":
            subprocess.Popen(f'start cmd /K python "{script_path}" monitor {self._interval} {self._threshold}', shell=True)
        elif platform.system() == "Linux":
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'python3 "{script_path}" monitor {self._interval} {self._threshold}; exec bash'])
        else:
            print("Unsupported platform for automatic terminal launching.")

    def run(self):
        """
        The method that starts the memory monitoring process.
        """
        while not self._stop_event.is_set():
            try:
                self.show_memory_usage()

                time.sleep(self._interval)
            except Exception as e:
                print(f"Error occurred in memory monitor: {e}")
                traceback.print_exc()

    def show_memory_usage(self):
        """
        Show memory instance
        """
        process = psutil.Process()
        tracemalloc.start()

        try:
            memory_info = process.memory_info()
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('traceback')

            # Display basic memory usage info
            print(f"{'Memory Monitor':^80}")
            print("=" * 80)
            print(f"{'RSS Memory Usage:':<40} {memory_info.rss / 1024 ** 2:.2f} MB")
            print("=" * 80)

            # Show the top memory consumers (files and lines of code)
            print(f"{'Top Memory Consumers':^80}")
            print(f"{'Size (MB)':<20}{'File':<40}{'Line Number':<20}{'Code':<40}")
            print("-" * 80)
            for stat in top_stats[:10]:
                frame = stat.traceback[0]
                filename = frame.filename
                lineno = frame.lineno

                # Read the actual line of code from the file
                code_line = self._get_code_line(filename, lineno)
                print(f"{stat.size / (1024 ** 2):<20.2f}{filename:<40}{lineno:<20}{code_line:<40}")
            print("=" * 80)

            # Track Large Objects and their referrers
            unreachable_objects = gc.collect()
            print(f"Unreachable objects: {unreachable_objects}")
            print(f"Remaining garbage: {gc.garbage}")
            big_objects = [obj for obj in gc.get_objects() if self._get_object_size(obj) > self._threshold]
            big_objects = sorted(big_objects, key=lambda obj: self._get_object_size(obj), reverse=True)[:5]

            print(f"{'Found Top 5 Big Objects':^80}")
            print(f"{'Object Type':<30}{'Size (MB)':<15}{'Thread Name':<20}{'Variable/Referrer':<20}")
            print("-" * 80)
            for obj in big_objects:
                thread_name = threading.current_thread().name

                # Get referrers (variables/objects holding this large object)
                referrers = gc.get_referrers(obj)
                referrer_names = []
                for ref in referrers:
                    if isinstance(ref, dict):
                        for key, value in ref.items():
                            if value is obj and isinstance(key, str):
                                referrer_names.append(key)

                referrer_str = ', '.join(referrer_names) if referrer_names else self._get_variable_name(obj)

                print(
                    f"{str(type(obj)):<30}{self._get_object_size(obj) / (1024 ** 2):.2f} MB {thread_name:<20}{referrer_str:<20}")

            print("=" * 80)

            # Track Circular References with Extended Objgraph
            print(f"{'Circular References (Top 3)':^80}")
            print("-" * 80)
            self._show_circular_references()
            print("-" * 80)

            # Detect and show backreferences (objects preventing garbage collection)
            print(f"{'Backrefs for Largest Objects':^80}")
            print("-" * 80)
            for obj in big_objects:
                objgraph.show_backrefs([obj], max_depth=3, too_many=10)  # Limit depth to 3 to avoid overload
                print("-" * 80)
        except Exception as e:
            print(f"Error occurred in memory monitor: {e}")
            traceback.print_exc()

    def _get_variable_name(self, obj):
        """
        Attempts to retrieve the variable name of the object by inspecting stack frames.
        This method looks into the local variables of stack frames and returns the first matching variable name.
        """
        frames = sys._current_frames()  # Get all current frames
        for thread_id, frame in frames.items():
            while frame:
                for var_name, var_value in frame.f_locals.items():
                    if var_value is obj:
                        return var_name  # Return the variable name if the value matches the object
                frame = frame.f_back  # Go to the previous frame
        return '<unknown>'  # If no variable name is found, return '<unknown>'

    def _show_circular_references(self):
        """
        Shows concise information about circular references: variable name, method, thread, and size.
        """
        obj_types = objgraph.most_common_types(limit=3)  # Limit to the 3 most common types

        for obj_type, count in obj_types:
            print(f"Type: {obj_type}, Count: {count}")
            sample_objects = objgraph.by_type(obj_type)

            # Limit to 1 object per type for conciseness
            for obj in sample_objects[:1]:
                size = self._get_object_size(obj) / (1024 ** 2)  # Convert size to MB

                # Attempt to find the variable name and thread
                referrers = gc.get_referrers(obj)
                if referrers:
                    for ref in referrers:
                        if isinstance(ref, dict):  # Check for dictionary (usually holds local variables)
                            for key, value in ref.items():
                                if value is obj:
                                    print(f"  Variable: {key}, Size: {size:.2f} MB")
                                    break

                # Print thread and method context
                print(f"  Thread Name: {threading.current_thread().getName()}")
                traceback_frames = sys._current_frames().get(threading.get_ident())
                if traceback_frames:
                    frame = traceback_frames.f_back
                    if frame:
                        method_name = frame.f_code.co_name
                        print(f"  Method: {method_name}")

    @staticmethod
    def _get_code_line(filename, lineno):
        """
        Gets the line of code from the file at the specified line number.

        Args:
            filename (str): The file from which to read the line.
            lineno (int): The line number to read.

        Returns:
            str: The line of code, or '<no code>' if the file or line cannot be read.
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                return lines[lineno - 1].strip() if lineno <= len(lines) else '<no code>'
        except Exception:
            return '<no code>'

    @staticmethod
    def _get_object_size(obj):
        """
        Returns the size of an object in bytes. For containers like lists and dictionaries,
        it sums the size of the container and its elements.

        Args:
            obj: The object to calculate the size of.

        Returns:
            int: The size of the object in bytes, or 0 in case of an error.
        """
        try:
            # Skip weak references and weak proxies without dereferencing
            if isinstance(obj, weakref.ProxyType):
                return 0  # Skip weak proxies

            if isinstance(obj, weakref.ReferenceType):
                obj = obj()  # Dereference the weakref
                if obj is None:
                    return 0  # If weakref is dead, return 0 size

            size = sys.getsizeof(obj)
            if isinstance(obj, (list, tuple, dict, set)):
                size += sum(map(sys.getsizeof, obj))
            return size
        except Exception as e:
            print(f"Error calculating size for object {type(obj)}: {e}")
            traceback.print_exc()
            return 0

    def _signal_handler(self, signum, frame):
        """
        Signal handler to stop the memory monitor when the process receives a termination signal.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self._stop_event.set()
        print(f"Signal {signum} received, stopping memory monitor.")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == 'monitor':
        interval = int(sys.argv[2])
        threshold = int(sys.argv[3])

        # Create the monitor instance using the passed arguments
        memory_monitor = CMemoryMonitor(interval=interval, threshold=threshold)
        memory_monitor.run()
