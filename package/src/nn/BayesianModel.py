import json
import os
import shutil

import tensorflow as tf
from src.distributions.Distribution import Distribution
from src.distributions.DistributionSerializer import DistributionSerializer


class BayesianModel:
    """
        a class that represents a trained bayesian model that could predict outputs. It could also be stored and loaded.
    """
    def __init__(self, model_config: dict):
        self._model_config = model_config
        # TODO : make it more general
        model: tf.keras.Model = tf.keras.Sequential().from_config(model_config)
        self._layers: list[BayesianModel.Layer] = []
        for layer in model.layers:
            self._layers.append(BayesianModel.Layer([w.shape for w in layer.get_weights()]))
        self._n_layers = len(self._layers)
        self._layers_dtbn_intervals = []
        self._distributions: list[Distribution] = []
        self._model: tf.keras.Model = model


    # intended implementation: dtbn intervals takes the form like [[0,2],[3,4],[5,5],[6,6],[7,12]...]
    # interval may be single (equal) or double (unequal); distribution list each element correspond to an interval
    def apply_distribution(self, distribution: Distribution, start_layer: int, end_layer: int):
        """
            set the distributions of the given layers to the one provided
            Args:
                distribution (Distribution): the distribution
                start_layer (int): the starting layer to be set
                end_layer (int): the ending layer to be set
        """
        if start_layer > end_layer:
            raise ValueError('starting_layer must be less than end_layer')
        elif start_layer < 0 or end_layer >= self._n_layers:
            raise ValueError('out of bounds')
        interval = [start_layer, end_layer]

        if len(self._layers_dtbn_intervals) == 0:
            self._layers_dtbn_intervals.append(interval)
            self._distributions.append(distribution)
            return
        for i in range(len(self._layers_dtbn_intervals)):
            if start_layer > self._layers_dtbn_intervals[i][0]:
                self._layers_dtbn_intervals = self._layers_dtbn_intervals[:i + 1] + [
                    interval] + self._layers_dtbn_intervals[i + 1:]
                self._distributions = self._distributions[:i + 1] + [distribution] + self._distributions[i + 1:]
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

    def apply_distributions_layers(self, layer_list, dtbn_list):
        """
        set the correlated layers and their distributions
        Args:
            layer_list (_type_): _description_
            dtbn_list (_type_): _description_
        """
        self._layers_dtbn_intervals = layer_list
        self._distributions = dtbn_list

    def _sample_weights(self):
        """
            sample the weights of the model
        """
        for interval_idx in range(len(self._layers_dtbn_intervals)):
            vector_weights: tf.Tensor = self._distributions[interval_idx].sample()
            start = self._layers_dtbn_intervals[interval_idx][0]
            end = self._layers_dtbn_intervals[interval_idx][1]
            take_from = 0
            for layer_idx in range(start, end + 1):
                weights = []
                for w in self._model.layers[layer_idx].get_weights():
                    size = tf.size(w).numpy()
                    weights.append(tf.reshape(vector_weights[take_from:take_from + size], w.shape))
                    take_from += size
                self._model.layers[layer_idx].set_weights(weights)


    def predict(self, x: tf.Tensor, nb_samples: int):
        """        
        use monte carlo approxiamtion over nb_samples to predict the result for the input
        

        Args:
            x (tf.Tensor): the inuput 
            nb_samples (int): number of samples for monte carlo approximation

        Returns:
            (list, tf.Tensor): a pair containing the sampled results and their mean as the prediction
        """
        result = 0
        samples_results = []
        for i in range(nb_samples):
            self._sample_weights()
            prediction = self._model(x)
            result += prediction
            samples_results.append(prediction)
        result /= nb_samples
        return samples_results, result

    @classmethod
    def load(cls, model_path: str, custom_distribution_register=None) -> 'BayesianModel':
        """
        load a model from a document
        Args:
            model_path (str): the path where the model is stored
            custom_distribution_register (dict, optional): If the distribution used is not implemented, 
            a deserialiser should be specified in this dctionnary. Defaults to None.

        Returns:
            BayesianModel: the loaded bayesian model
        """
        if custom_distribution_register is None:
            custom_distribution_register = dict()
        with open(os.path.join(model_path, "config.json"), "r") as config_file:
            
            bayesian_model = BayesianModel(tf.keras.models.model_from_json(config_file.read()).get_config())

        layers_intervals = []
        with open(os.path.join(model_path,"layers_config.txt"), "r") as layers_file:
            # layers information are saved as name, start, end (both included)
            n_intervals = int(layers_file.readline())
            for i in range(n_intervals):
                layers_intervals.append(
                    (layers_file.readline()[:-1], int(layers_file.readline()), int(layers_file.readline())))

        for i in range(n_intervals):
            distribution = DistributionSerializer.load_from(
                layers_intervals[i][0], 
                custom_distribution_register,
                os.path.join(model_path,"distribution" + str(i), "distribution.json")
            )
            bayesian_model.apply_distribution(distribution, layers_intervals[i][1], layers_intervals[i][2])
        return bayesian_model
    
    def _empty_folder(self,path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def store(self, model_path: str):
        """
        store a model to a document
        Args:
            model_path (str): the path where the model will be stored, this folder should already exist on the system
            custom_distribution_register (dict, optional): If the distribution used is not implemented, 
            a serialiser should be specified in this dctionnary. Defaults to None.

        Returns:
            BayesianModel: the loaded bayesian model
        """
        if os.path.exists(model_path) == False:
            raise Exception("inexisting directory: " + model_path) 
        self._empty_folder(model_path)
        with open(os.path.join(model_path,"config.json"), "w") as config_file:
            config_file.write(self._model.to_json())

        with open(os.path.join(model_path, "layers_config.txt"), "w") as layers_file:
            layers_file.write(str(len(self._layers_dtbn_intervals))+'\n')
            for i in range(len(self._layers_dtbn_intervals)):
                layers_start = self._layers_dtbn_intervals[i][0]
                layers_end = self._layers_dtbn_intervals[i][1]
                layers_file.write(self._distributions[i].__class__.__name__+'\n'+str(layers_start)+'\n'+str(layers_end)+'\n')

        for i in range(len(self._layers_dtbn_intervals)):
            os.mkdir(os.path.join(model_path, "distribution"+str(i)))
            self._distributions[i].store(os.path.join(model_path, "distribution"+str(i),"distribution.json"))

    class Layer:
        def __init__(self, weights_shape: list[tf.TensorShape]):
            self.weights_shape = weights_shape
            self._n_params = 0
            for w in weights_shape:
                self._n_params += tf.math.reduce_sum(w).numpy()

        @property
        def n_params(self) -> int:
            return self._n_params
