from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution
from src.nn.BayesianModel import BayesianModel
from src.optimizers.Optimizer import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp


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

    def step(self):
        sample, label = next(self._data_iterator)
        self._base_model.fit(sample, label, epochs=1, batch_size=1, verbose=0)
        bayesian_layer_index = 0
        for layer_index in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_index]
            if len(layer.get_weights()) != 0:
                theta = [tf.reshape(i, (-1, 1)) for i in layer.get_weights()]
                theta = tf.reshape(tf.concat(theta, 0), (-1,))
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
        self._k = self._hyperparameters["k"]
        self._frequency = self._hyperparameters["frequency"]
        self._scale = self._hyperparameters["scale"]
        self._lr = self._hyperparameters["lr"]
        self._base_model = tf.keras.models.clone_model(kwargs["starting_model"])
        self._dataloader = (self._dataset.tf_dataset()
                            .shuffle(self._dataset.tf_dataset().cardinality())
                            .batch(1))
        self._base_model_optimizer = tf.keras.optimizers.SGD(learning_rate=self._hyperparameters.lr)
        self._base_model.compile(self._base_model_optimizer, self._dataset.loss())
        self._init_swag_arrays()
        self._data_iterator = iter(self._dataloader)
        self._n = 0

    def _init_swag_arrays(self):
        for layer in list(self._base_model.layers):
            size = 0
            for w in layer.get_weights():
                size += tf.size(w).numpy()
            if size != 0:
                self._mean.append(tf.zeros((size,), dtype=tf.float32))
                self._sq_mean.append(tf.zeros((size,), dtype=tf.float32))
                self._dev.append(tf.zeros((size, 0), dtype=tf.float32))

    def result(self) -> BayesianModel:
        model = BayesianModel(self._model_config)
        i = 0
        for mean, sq_mean, dev in zip(self._mean, self._sq_mean, self._dev):
            tf_dist = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
                mean,
                sq_mean - mean ** 2,
                dev
            )
            model.apply_distribution(TensorflowProbabilityDistribution(tf_dist), i, i)
            i += 1
        return model

    def update_parameters_step(self):
        pass
