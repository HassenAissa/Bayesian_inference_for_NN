from math import sqrt
import os

from PyAce.distributions import MultivariateNormalDiagPlusLowRank
from PyAce.distributions.tf import TensorflowProbabilityDistribution
from PyAce.nn import BayesianModel
from . import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import copy
import numpy as np

# lr_func kwarg represents the learning rate for each step
class SGLD(Optimizer):
    def __init__(self):
        super().__init__()
        self._n = None
        self._data_iterator = None
        self._dataloader = None
        self._base_model_optimizer = None
        self._base_model: tf.keras.Model = None
        self._lr_upper = None
        self._lr_lower = None
        self._lr_gamma = None
        self._lr = None
        self._mean: list[tf.Tensor] = []
        self._sum = 0
        self._sq_mean: list[tf.Tensor] = []
        self._dev: list[tf.Tensor] = []
        self._weight_layers_indices = []

    def step(self, save_document_path = None):
        # get the sample and the label
        sample,label = next(self._data_iterator, (None,None))
        # if the iterator reaches the end of the dataset, reinitialise the iterator
        if sample is None:
            self._data_iterator = iter(self._dataloader)
            sample, label = next(self._data_iterator, (None, None))

        with tf.GradientTape(persistent=True) as tape:
            predictions = self._base_model(sample, training = True)
            # get the loss
            loss = self._dataset.loss()(label, predictions)
            # save the loss if the path is specified
            if save_document_path != None:
                with open(save_document_path, "a") as losses_file:
                    losses_file.write(str(loss.numpy()))

        var_grad = tape.gradient(loss, self._base_model.trainable_variables)
        for var, grad in zip(self._base_model.trainable_variables, var_grad):
            if grad is not None:
                noise = tf.random.normal(shape=grad.shape, mean = 0.0, stddev=self._lr(self._n))
                var.assign_add(-self._lr(self._n) * (grad + noise)) 

        bayesian_layer_index = 0
        for layer_index in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_index]

            if len(layer.trainable_variables) != 0:
                theta = [tf.reshape(i, (-1, 1)) for i in layer.trainable_variables]
                theta = tf.reshape(tf.concat(theta, 0), (-1, 1))
                
                mean = self._mean[bayesian_layer_index]
                sq_mean = self._sq_mean[bayesian_layer_index]

                # update the mean
                mean = (mean * self._n + theta) / (self._n + 1.0)
                self._mean[bayesian_layer_index] = mean

                # update the second moment
                sq_mean = (sq_mean * self._n + theta ** 2) / (self._n + 1.0)
                self._sq_mean[bayesian_layer_index] = sq_mean

                # update the deviation matrix
                deviation_matrix = self._dev[bayesian_layer_index]
                self._dev[bayesian_layer_index] = tf.concat(
                    (deviation_matrix, theta - mean), axis=1)
                bayesian_layer_index += 1
        self._n += 1
        return self._running_loss / self._n        
        
    def _init_arrays(self):
        """
        initialise arrays to keep track of mean, sq_mean, standard deviation
        """
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            size = 0
            for w in layer.trainable_variables:
                size += tf.size(w).numpy()
            if size != 0:
                self._mean.append(tf.zeros((size, 1), dtype=tf.float32))
                self._sq_mean.append(tf.zeros((size, 1), dtype=tf.float32))
                self._dev.append(tf.zeros((size, 0), dtype=tf.float32))
                self._weight_layers_indices.append(layer_idx)

    def _init_sgld_lr(self):
        n = self._nb_iterations
        l_g = np.power(self._lr_lower, 1.0 / self._lr_gamma)
        u_g = np.power(self._lr_upper, 1.0 / self._lr_gamma)
        b = -(n * l_g) / (l_g - u_g)
        a = self._lr_upper * np.power(b, self._lr_gamma)
        self._lr = lambda step: a * np.power((b+step), -self._lr_gamma)

    def train(self, nb_iterations: int, loss_save_document_path: str = None, model_save_frequency: int = None,
              model_save_path: str = None, weights_and_biases_log = False):
        self._nb_iterations = nb_iterations
        self._init_sgld_lr()
        super().train(nb_iterations, loss_save_document_path, model_save_frequency, model_save_path, weights_and_biases_log)

    def update_parameters_step(self):
        return super().update_parameters_step()
        
    def compile_extra_components(self, **kwargs):
        """
            compiles components of subclasses
        """
        self._batch_size = int(self._hyperparameters.batch_size)
        self._lr_upper = self._hyperparameters.lr[0]
        self._lr_lower = self._hyperparameters.lr[1]
        self._lr_gamma = self._hyperparameters.lr[2]
        self._base_model = tf.keras.models.model_from_json(self._model_config)
        self._dataset_setup()
        self._init_arrays()
        self._n = 0
        self._running_loss = 0

    def result(self) -> BayesianModel:
        model = BayesianModel(self._model_config)
        for mean, sq_mean, dev, idx in zip(self._mean, self._sq_mean, self._dev,
                                           range(len(self._weight_layers_indices))):
            # tf.debugging.check_numerics(dev, "dev")
            # tf.debugging.check_numerics(mean, "mean")
            # tf.debugging.check_numerics(sq_mean, "sq_meqn")

            tf_dist = tfp.distributions.Normal(
                tf.reshape(mean, (-1,)),
                tf.reshape(sq_mean - mean ** 2, (-1,))
            )

            tf_dist = TensorflowProbabilityDistribution(
                tf_dist
            )
            start_idx = self._weight_layers_indices[idx]
            end_idx = len(self._base_model.layers) - 1
            if idx + 1 < len(self._weight_layers_indices):
                end_idx = self._weight_layers_indices[idx + 1]

            model.apply_distribution(tf_dist, start_idx, start_idx)
        return model
        
