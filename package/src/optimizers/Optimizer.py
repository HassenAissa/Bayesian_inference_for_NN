from abc import ABC, abstractmethod

from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
import tensorflow as tf
import os
import shutil

class Optimizer(ABC):

    def __init__(self):
        self._model_config = None
        self._hyperparameters = None
        self.__compiled = False
        self._dataset : Dataset = None

    @abstractmethod
    def step(self, save_document_path = None):
        pass

    def compile(self, hyperparameters: HyperParameters, model_config: dict,dataset, **kwargs):
        if self.__compiled:
            raise Exception("Model Already compiled")
        else:
            self.__compiled = True
            self._hyperparameters = hyperparameters
            self._model_config = model_config
            self._dataset = dataset
        self.compile_extra_components(**kwargs)

    @abstractmethod
    def compile_extra_components(self, **kwargs):
        pass

    @abstractmethod
    def update_parameters_step(self):
        pass

    def train(self, nb_iterations, loss_save_document_path = None, model_save_frequency = None, model_save_path = None):
        if model_save_frequency == None and model_save_path != None:
            raise Exception("Error: save path precised and save frequency is None, please provide a savong frequency")
        if model_save_frequency != None and model_save_path == None:
            raise Exception("Error: save frequency precised and save path is None, please provide a saving path")

        if loss_save_document_path != None and os.path.exists(loss_save_document_path):
            os.remove(loss_save_document_path)

        
        saved_model_nbr = 0
        for i in range(nb_iterations):
            self.step(loss_save_document_path)
            if model_save_frequency != None and i%model_save_frequency == 0:
                bayesian_model = self.result()
                if os.path.exists(os.path.join(model_save_path, "model"+str(saved_model_nbr))):
                    shutil.rmtree(os.path.join(model_save_path, "model"+str(saved_model_nbr)))
                os.makedirs(os.path.join(model_save_path, "model"+str(saved_model_nbr)))               
                bayesian_model.store(os.path.join(model_save_path, "model"+str(saved_model_nbr)))
                saved_model_nbr += 1
                # if int(i/nb_iterations *100) > int((i-1)/nb_iterations *100):
            print(" Training in progress... \r{} %".format(int(i/nb_iterations *100)), end = '')
        print(" Training in progress... \r{} %".format(100), end = '')
        print()

    @abstractmethod
    def result(self) -> BayesianModel:
        pass

