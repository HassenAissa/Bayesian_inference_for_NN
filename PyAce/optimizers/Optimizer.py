import math
from abc import ABC, abstractmethod

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from .hyperparameters import HyperParameters
import tensorflow as tf
import os
import shutil
import wandb
from wandb.keras import WandbCallback


class Optimizer(ABC):

    def __init__(self):
        self._model_config = None
        self._hyperparameters = None
        self.__compiled = False
        self._dataset: Dataset = None

    @abstractmethod
    def step(self, save_document_path=None):
        """
        TODO : Add loss return signature
        Performs one step of the training

        Args:
            save_document_path (_type_, optional): The path to save the losses during the training. Defaults to None.

        Returns:
            float: the loss value after the step
        """
        pass

    def dataset_setup(self):
        self._training_dataset: tf.data.Dataset = self._dataset.training_dataset()
        self._training_dataset_cardinality = self._training_dataset.cardinality().numpy().item()
        self._dataloader = (self._training_dataset
                            .shuffle(self._training_dataset_cardinality)
                            .batch(self._batch_size))
        self._data_iterator = iter(self._dataloader)
        

    def compile(self, hyperparameters: HyperParameters, model_config: str, dataset, verbose=True,**kwargs):
        """compile the model

        Args:
            hyperparameters (HyperParameters): the model hyperparameters
            model_config (dict): the configuration of the model
            dataset (_type_): the dataset of the model

        Raises:
            Exception: raises error if the model is already compiled
        """
        if self.__compiled:
            raise Exception("Model Already compiled")
        else:
            self.__compiled = True
            self._hyperparameters = hyperparameters
            self._model_config = model_config
            self._dataset = dataset
            self._verbose = verbose
        self.compile_extra_components(**kwargs)

    @abstractmethod
    def compile_extra_components(self, **kwargs):
        """
        used to compile components of subclasses
        """
        pass

    @abstractmethod
    def update_parameters_step(self):
        """
        one step of updating the model parameters
        """
        pass

    def _empty_folder(self, path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def train_with_weights_and_biases(self, nb_iterations, project_name, weights_and_biases_config):
        wandb.login()
        run = wandb.init(project= project_name, config=weights_and_biases_config)
        self.train(nb_iterations, weights_and_biases_log = True)

    def train(self, nb_iterations: int, loss_save_document_path: str = None, model_save_frequency: int = None,
              model_save_path: str = None, weights_and_biases_log = False):
        """
        trains the model and saved the training metrics and model status

        Args:
            nb_iterations (int): number of training iterations
            loss_save_document_path (str, optional): The path to save the loss during training. Defaults to None.
            model_save_frequency (int, optional): The frequency of saving the models during training. Defaults to None.
            model_save_path (str, optional): The path to save the models during training. Defaults to None.

        Raises:
            Exception: if the model saving path is specified and the frequency of saving the model is not, or
            if the frequency of saving the model is sprecified and the model saving path is not.
        """
        if model_save_frequency == None and model_save_path != None:
            raise Exception("Error: save path precised and save frequency is None, please provide a savong frequency")
        if model_save_frequency != None and model_save_path == None:
            raise Exception("Error: save frequency precised and save path is None, please provide a saving path")

        if loss_save_document_path != None and os.path.exists(loss_save_document_path):
            os.remove(loss_save_document_path)

        if model_save_path != None:
            self._empty_folder(model_save_path)            


        saved_model_nbr = 0
        for i in range(nb_iterations):
            loss = self.step(loss_save_document_path)
            self._print_progress(i / nb_iterations, loss=loss)
            if weights_and_biases_log == True:
                wandb.log({
                    "loss": loss
                })
            if model_save_frequency != None and i % model_save_frequency == 0:
                bayesian_model = self.result()
                if os.path.exists(os.path.join(model_save_path, "model" + str(saved_model_nbr))):
                    shutil.rmtree(os.path.join(model_save_path, "model" + str(saved_model_nbr)))
                os.makedirs(os.path.join(model_save_path, "model" + str(saved_model_nbr)))
                bayesian_model.store(os.path.join(model_save_path, "model" + str(saved_model_nbr)))
                saved_model_nbr += 1

        print()

    @abstractmethod
    def result(self) -> BayesianModel:
        """
        create a bayesian model at the stage of the training

        Returns:
            BayesianModel: the bayesian model trained
        """
        pass

    def _print_progress(self, progress: float, bar_length=10, suffix="Training", **kwargs):
        if not self._verbose:
            return
        nb_chars = math.ceil(progress * bar_length)
        bar = "[" + nb_chars * "="
        if nb_chars < bar_length:
            bar += ">"
        bar += "]"
        infos = ' '.join("{}: {}".format(k, v) for k, v in kwargs.items())
        percentage = str(math.ceil(progress * 100))
        print("\r" + suffix + " " + percentage + " % " + bar + " " + infos, end="")

    def _new_progress_line(self):
        if not self._verbose:
            return
        print()