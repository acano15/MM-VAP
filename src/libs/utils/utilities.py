import functools
from threading import Thread
import concurrent.futures
import psutil
from typing import Any


def threaded_method(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def task():
            return func(*args, **kwargs)

        if psutil.cpu_percent() >= 90:  # Adjust the threshold as needed
            return func(*args, **kwargs)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the task to the executor
            future = executor.submit(task)
            # Return the future object instead of waiting for the result
            return future
    return wrapper


def threaded_method_join(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        def task():
            return func(*args, **kwargs)

        if psutil.cpu_percent() >= 90:  # Adjust the threshold as needed
            return func(*args, **kwargs)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the task to the executor
            future = executor.submit(task)
            # Wait for the task to complete and retrieve the result
            result = future.result()
        return result
    return wrapper


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None

    def run(self):
        """
        Run
        """
        if self._Thread__target is not None:
            self._return = self._Thread__target(
                *self._Thread__args,
                **self._Thread__kwargs)

    def join(self) -> Any:
        """
        Join
        """
        Thread.join(self)
        return self._return
