import math
from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Type
import multiprocessing

class HyperparameterOptimizer(ABC):
    """
    An abstarct class that represents a hyperparameter optimizer

    """
    def __init__(self):
        self._f = None

    def compile(self, f, *args, **kwargs):
        """compiles the HyperparametersOptimizer instance with its specific extra components.

        Args:
            f (_type_): _description_ TODO
        """
        self._f = f
        self._compile_extra_components(*args, **kwargs)

    @abstractmethod
    def _compile_extra_components(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, n_processes: int):
        """
        This function should define the optimization policy in inheritant classes.

        Args:
            n_processes (int): the number of processes with different combinations of hyperparameters
        """
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
