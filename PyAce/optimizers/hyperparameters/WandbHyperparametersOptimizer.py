import copy
from functools import cmp_to_key
import math
from PyAce.datasets import Dataset
from .. import Optimizer
from . import HyperParameters
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback


class WandbHyperparametersOptimizer:
    """
        This class allows performing hyperparameter optimization using Weights and Biases.
        Args:
            dataset (Dataset): The dataset on which we perform the hyperparameter tuning.
            optimizer_class (Optimizer): The inference method to use
            model_config (str): The json of the model on which to train
            **kwargs: the kwargs are the extra compiling arguments for the Optimizer
    """
    def __init__(self, dataset: Dataset, optimizer_class: Optimizer, model_config: str, **kwargs):

        self._additional_compile_params = copy.deepcopy(kwargs)
        self._dataset = dataset
        self._model_config = model_config
        self._hyperparameters_list = []
        self._optimizer_class = optimizer_class

    def __getattr__(self, item):
        if item in self._params:
            return self._params[item]
        else:
            raise AttributeError("'HyperParameters' object has no attribute " + str(item))
           
    def _sweep_train(self, config_defaults=None):
        config_defaults = {}
        wandb.init(config=config_defaults, name = "")
        optimizer = self._optimizer_class()
        hyperparameters = HyperParameters(**wandb.config)
        optimizer.compile(hyperparameters, self._model_config, self._dataset, **self._additional_compile_params)
        optimizer.train(wandb.config.epochs, weights_and_biases_log = True)

        

    def hyper_parameter_tuning_with_weights_and_biases(self, sweep_config: dict, project_name: str = "", count: int = 50):
        """
        Tunes the hyperparameters using Weights and Biases library

        Args:
            sweep_config (dict): the Wights and Biases config. Please refer to this link for an example: \n \
            https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
            project_name (str, optional): The project name, if absent a random one will be assigned by Weights and Biases. Defaults to "".
            count (int, optional): Maximum number of runs. Defaults to 50.
        """
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=self._sweep_train, count= count)
