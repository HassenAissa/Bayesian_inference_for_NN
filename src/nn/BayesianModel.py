import tensorflow as tf
from src.distributions.Distribution import Distribution
import bisect


class BayesianModel:
    def __init__(self, model_config: dict):
        self._model_config = model_config
        model: tf.keras.Model = tf.keras.models.model_from_config(model_config)
        self._layers: list[BayesianModel.Layer] = []
        for layer in model.layers:
            self._layers.append(BayesianModel.Layer([w.shape for w in layer.weights]))
        self._n_layers = len(self._layers)
        self._layers_dtbn_intervals = []
        self._distributions = []
        self._model = model

    def apply_distribution(self, distribution: Distribution, start_layer: int, end_layer: int):
        # intended implementation: dtbn intervals takes the form like [[0,2],[3,4],[5],[6],[7,12]...]
        # interval may be single (equal) or double (unequal); distribution list each element correspond to an interval
        if start_layer > end_layer:
            raise ValueError('starting_layer must be less than end_layer')
        elif start_layer < 0 or end_layer >= self._n_layers:
            raise ValueError('out of bounds')
        interval = [start_layer]
        if end_layer > start_layer:
            interval.append(end_layer)
        for i in range(len(self._layers_dtbn_intervals)):
            if start_layer > self._layers_dtbn_intervals[i][0]:
                self._layers_dtbn_intervals = self._layers_dtbn_intervals[:i+1]+[interval]+self._layers_dtbn_intervals[i+1:]
                self._distributions = self._distributions[:i+1]+[distribution]+self._distributions[i+1:]
                break
        """
        TODO: Implement this without errors
        if start_layer > end_layer:
            raise ValueError('starting_layer must be less than end_layer')
        elif start_layer < 0 or end_layer >= self._n_layers:
            raise ValueError('out of bounds')
        start_index = bisect.bisect_right(self._layers_dtbn_intervals, start_layer)
        if self._layers_dtbn_intervals[start_index - 1] != start_layer:
            self._layers_dtbn_intervals.insert(start_index, start_layer)
        else:
            start_index -= 1
        end_index = bisect.bisect_right(end_layer)
        if self._layers_dtbn_intervals[end_index]:
            pass
        """
        pass

    def apply_distributions_layers(self, layer_list, dtbn_list):
        self._layers_dtbn_intervals = layer_list
        self._distributions = dtbn_list

    def _sample_weights(self):
        pass

    def predict(self, x: tf.Tensor):
        pass

    @classmethod
    def load(cls, model_path: str) -> 'BayesianModel':
        pass

    def store(self, model_path: str):
        pass

    class Layer:
        def __init__(self, weights_shape: list[tf.TensorShape]):
            self.weights_shape = weights_shape
            self._n_params = 0
            for w in weights_shape:
                self._n_params += tf.math.reduce_sum(w).numpy()

        @property
        def n_params(self) -> int:
            return self._n_params
