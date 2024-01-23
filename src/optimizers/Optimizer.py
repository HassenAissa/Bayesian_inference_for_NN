from abc import ABC, abstractmethod

from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
import tensorflow as tf

class Optimizer(ABC):

    def __init__(self):
        self._model_config = None
        self._hyperparameters = None
        self.__compiled = False

    @abstractmethod
    def step(self):
        pass

    def compile(self, hyperparameters: HyperParameters, model_config: dict,dataset, **kwargs):
        if self.__compiled:
            raise Exception("Model Already compiled")
        else:
            self.__compiled = True
            self._hyperparameters = hyperparameters
            self._model_config = model_config
        self.compile_extra_components(**kwargs)

    @abstractmethod
    def compile_extra_components(self, **kwargs):
        pass

    @abstractmethod
    def update_parameters_step(self):
        pass

    def train(self):
        """
        TODO : Add some default training method that calls step() a lot of times
        Returns:

        """
        pass
    @abstractmethod
    def result(self) -> BayesianModel:
        pass

