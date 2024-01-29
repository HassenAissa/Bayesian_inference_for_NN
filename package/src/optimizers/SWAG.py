from math import sqrt

from src.distributions.MultivariateNormalDiagPlusLowRank import MultivariateNormalDiagPlusLowRank
from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution
from src.nn.BayesianModel import BayesianModel
from src.optimizers.Optimizer import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import copy


class SWAG(Optimizer):

    def __init__(self):
        super().__init__()
        self._n = None
        self._data_iterator = None
        self._dataloader = None
        self._base_model_optimizer = None
        self._base_model: tf.keras.Model = None
        self._lr = None
        self._scale = None
        self._frequency = None
        self._k = None
        self._mean: list[tf.Tensor] = []
        self._sq_mean: list[tf.Tensor] = []
        self._dev: list[tf.Tensor] = []
        self._weight_layers_indices = []

    def step(self):
        sample,label = next(self._data_iterator, None)
        if sample is None:
            self._data_iterator = iter(self._dataloader)
            sample, label = next(self._data_iterator, (None, None))
        with tf.GradientTape(persistent=True) as tape:
            predictions = self._base_model(sample, training = True)
            loss = self._dataset.loss()(label, predictions)
        weight_gradient = tape.gradient(loss, self._base_model.trainable_variables)
        weights = self._base_model.get_weights()
        new_weights = []
        for i in range(len(weight_gradient)):
            wg = tf.math.multiply(weight_gradient[i], self._lr)
            m = tf.math.subtract(weights[i], wg)
            new_weights.append(m)
        self._base_model.set_weights(new_weights)
        bayesian_layer_index = 0
        for layer_index in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_index]

            if len(layer.get_weights()) != 0:
                theta = [tf.reshape(i, (-1, 1)) for i in layer.get_weights()]
                theta = tf.reshape(tf.concat(theta, 0), (-1, 1))
                if self._n % self._hyperparameters.frequency == 0:
                    mean = self._mean[bayesian_layer_index]
                    sq_mean = self._sq_mean[bayesian_layer_index]
                    mean = (mean * self._n + theta) / (self._n + 1.0)
                    sq_mean = (sq_mean * self._n + theta ** 2) / (self._n + 1.0)
                    self._mean[bayesian_layer_index] = mean
                    self._sq_mean[bayesian_layer_index] = sq_mean

                    deviation_matrix = self._dev[bayesian_layer_index]

                    if deviation_matrix.shape[0] == self._hyperparameters.k:
                        self._dev[bayesian_layer_index] = tf.concat(
                            (deviation_matrix[:, :self._hyperparameters.k - 1], theta - mean), axis=1)
                    else:
                        self._dev[bayesian_layer_index] = tf.concat(
                            (deviation_matrix, theta - mean), axis=1)
                bayesian_layer_index += 1
        self._n += 1

    def compile_extra_components(self, **kwargs):
        self._k = self._hyperparameters.k
        self._frequency = self._hyperparameters.frequency
        self._scale = self._hyperparameters.scale
        self._lr = self._hyperparameters.lr
        self._base_model = tf.keras.models.clone_model(kwargs["starting_model"])
        self._base_model.set_weights(kwargs["starting_model"].get_weights())
        self._dataloader = (self._dataset.tf_dataset()
                            .shuffle(self._dataset.tf_dataset().cardinality())
                            .batch(1))
        self._init_swag_arrays()
        self._data_iterator = iter(self._dataloader)
        self._n = 0

    def _init_swag_arrays(self):
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            size = 0
            for w in layer.get_weights():
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
            tf.debugging.check_numerics(dev, "dev")
            tf.debugging.check_numerics(mean, "mean")
            tf.debugging.check_numerics(sq_mean, "sq_meqn")

            tf_dist = MultivariateNormalDiagPlusLowRank(
                tf.reshape(mean, (-1,)),
                tf.reshape(sq_mean - mean ** 2, (-1,)),
                sqrt((1 / (self._k - 1))) * dev,
            )

            '''
            tf_dist = tfp.distributions.MultivariateNormalDiag(
                loc = tf.reshape(mean, (-1,)),
                scale_diag=tf.reshape(sq_mean - mean ** 2, (-1,))

            )
            '''
            start_idx = self._weight_layers_indices[idx]
            end_idx = len(self._base_model.layers) - 1
            if idx + 1 < len(self._weight_layers_indices):
                end_idx = self._weight_layers_indices[idx + 1]

            model.apply_distribution(tf_dist, start_idx, start_idx)
        return model

    def update_parameters_step(self):
        pass
