import math
from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Type
import multiprocessing
from PyAce.optimizers import Optimizer

class HyperparameterOptimizer(ABC):
    def __init__(self):
        self._f = None

    def compile(self, f, *args, **kwargs):
        self._f = f
        self._compile_extra_components(*args, **kwargs)

    @abstractmethod
    def _compile_extra_components(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, n_processes: int):
        pass
    def _print_progress(self, progress: float, bar_length=10, suffix="Training", **kwargs):
        nb_chars = math.ceil(progress * bar_length)
        bar = "[" + nb_chars * "="
        if nb_chars < bar_length:
            bar += ">"
        bar += "]"
        infos = ' '.join("{}: {}".format(k, v) for k, v in kwargs.items())
        percentage = str(math.ceil(progress * 100))
        print("\r" + suffix + " " + percentage + " % " + bar + " " + infos, end="")
