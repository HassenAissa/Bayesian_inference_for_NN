from math import sqrt
import os

from Pyesian.distributions import MultivariateNormalDiagPlusLowRank
from Pyesian.distributions.tf import TensorflowProbabilityDistribution
from Pyesian.nn import BayesianModel
from . import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import copy


class SWAG(Optimizer):
    """
    SWAG is a class that inherits from Optimizer. \n
    This simple inference methods is taken from the paper : "A simple baseline for Bayesian uncertianty in deep learning". \n
    https://arxiv.org/pdf/1902.02476.pdf  \n
    This inference methods takes the following hyperparameters:
    Hyperparameters:
        `batch_size`: the size of the batch for one step. Defaults to 128. \n
        `lr`: the learning rate \n
        `k`: maximum number of columns in the deviation matrix. It should not be very big so that it takes into account 
        only the last part of the training where we start converging. Should be at least 2. \n
        `scale`: the scale of the deviation matrix. It should be between 0 and 1 \n
        `frequency`: moment update frequency. It could be left to 1 and increased for performance reasons. \n
    """

    def __init__(self):
        super().__init__()
        self._n = None
        self._data_iterator = None
        self._dataloader = None
        self._base_model_optimizer = None
        self._base_model: tf.keras.Model = None
        self._lr = None
        self._frequency = None
        self._k = None
        self._mean: list[tf.Tensor] = []
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
                var.assign_sub(self._lr * grad)  # assign_sub for SGD update

        bayesian_layer_index = 0
        for layer_index in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_index]

            if len(layer.trainable_variables) != 0:
                theta = [tf.reshape(i, (-1, 1)) for i in layer.trainable_variables]
                theta = tf.reshape(tf.concat(theta, 0), (-1, 1))
                if self._n % self._frequency == 0:
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
                    if deviation_matrix.shape[1] == self._k:
                        self._dev[bayesian_layer_index] = tf.concat(
                            (deviation_matrix[:, :self._k - 1], theta - mean), axis=1)
                    else:
                        self._dev[bayesian_layer_index] = tf.concat(
                            (deviation_matrix, theta - mean), axis=1)
                bayesian_layer_index += 1
        self._n += 1
        return loss


    def compile_extra_components(self, **kwargs):
        """
            compiles components of subclasses
            Args:
                starting_model: this is the starting model for the inference method. It could be a pretrained model.
        """
        self._k = int(self._hyperparameters.k)
        self._frequency = int(self._hyperparameters.frequency)
        self._lr = self._hyperparameters.lr
        self._scale = self._hyperparameters.scale
        self._base_model = tf.keras.models.clone_model(kwargs["starting_model"])
        self._base_model.set_weights(kwargs["starting_model"].get_weights())
        self._batch_size = int(self._hyperparameters.batch_size)
        self._dataset_setup()
        self._init_swag_arrays()
        self._n = 0

    def _init_swag_arrays(self):
        """
        initialise the mean, second moment (sq_mean), deviation and trainable weights lists
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

    def result(self) -> BayesianModel:
        model = BayesianModel(self._model_config)
        for mean, sq_mean, dev, idx in zip(self._mean, self._sq_mean, self._dev,
                                           range(len(self._weight_layers_indices))):
            # tf.debugging.check_numerics(dev, "dev")
            # tf.debugging.check_numerics(mean, "mean")
            # tf.debugging.check_numerics(sq_mean, "sq_meqn")
            tf_dist = MultivariateNormalDiagPlusLowRank(
                tf.reshape(mean, (-1,)),
                tf.reshape(sq_mean - mean ** 2, (-1,)),
                sqrt((self._scale / (self._k - 1))) * dev,
            )
            start_idx = self._weight_layers_indices[idx]
            end_idx = len(self._base_model.layers) - 1
            if idx + 1 < len(self._weight_layers_indices):
                end_idx = self._weight_layers_indices[idx + 1]

            model.apply_distribution(tf_dist, start_idx, start_idx)
        return model

    def update_parameters_step(self):
        pass
