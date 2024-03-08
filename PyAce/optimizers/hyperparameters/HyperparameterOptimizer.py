import math
from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Type
import multiprocessing

class HyperparameterOptimizer(ABC):
    """
    An abstract class that represents a hyperparameter optimizer

    """
    def __init__(self):
        self._f = None

    def compile(self, f, *args, **kwargs):
        """Provide the optimizer with the search space for the hyperparameters.
        compile should be called before calling optimize.

        Args:
            *args (list[Number]): list of hyperparameters of the model with name and range.
            **kwargs: additional arguments for child classes
            f (_type_): the cost function to optimize. You can train and evaluate your model on the hyperparameters here
        """
        self._f = f
        self._compile_extra_components(*args, **kwargs)

    @abstractmethod
    def _compile_extra_components(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self):
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
