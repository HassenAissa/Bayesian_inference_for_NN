from optimizer import Optimizer
import sys
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src')
from hyperparameters import Hyperparams
import copy
import utils
import math
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from collections import namedtuple
 

class SwagHyperparam(Hyperparams):
    def __init__(self, loss, k=20, frequency=5, scale = 1, lr = 1e-1):
        self.k = k
        self.frequency = frequency
        self.scale = scale
        self.lr = lr
        self.loss = loss


# class SwagWeight:
#     def __init__(self, mean, sq_mean, deviation_matrix):
#         self.mean = mean
#         self.sq_mean = sq_mean
#         self.deviation_matrix = deviation_matrix

class SWAG(Optimizer):

    def __init__(self,base_model, dataloader, hyperparameters):
        super(SWAG, self).__init__(hyperparameters)
        self.params = list()
        self.n_models = 1
        self.base_model = base_model
        self.swag_parameters()
        self.dataloader = iter(dataloader)
        self.hyperparameters = hyperparameters
        self.base_model_optimizer = tf.keras.optimizers.SGD(learning_rate=self.hyperparameters.lr)
        self.base_model.compile(self.base_model_optimizer, self.hyperparameters.loss)

    def swag_parameters(self):
        for layer in list(self.base_model.layers):
            layer_weights = layer.get_weights()
            swag_weights = []
            for i in range(len(layer_weights)):
                w = [tf.zeros_like(layer_weights[i]),
                    tf.zeros_like(layer_weights[i]),
                    tf.zeros([0, (tf.size(layer_weights[i])).numpy()])]
                
                swag_weights.append(w)
            self.params.append(swag_weights)

    def update_hyperparam():
        pass

    def hyperparam_init():
        pass

    def distribution(self):
        mean_list = []
        sq_mean_list = []
        deviation_list = []
        for bayesian_layers in self.params:
            for bayesian_weights in bayesian_layers:
                mean = bayesian_weights[0]
                sq_mean = bayesian_weights[1]
                mean_list.append(mean)
                sq_mean_list.append(sq_mean)
                deviation = bayesian_weights[2]
                deviation_list.append(deviation)

        mean = utils.flatten(mean_list)
        sq_mean = utils.flatten(sq_mean_list)
        deviation = tf.concat(deviation_list, axis=1)
        distribution = tfp.distributions.MultivariateNormalDiagPlusLowRankCovariance(
            mean,sq_mean - mean ** 2, tf.transpose(deviation))
        return distribution

        

    def predict(self, distribution, input):

        mean_list = []

        for bayesian_layers in self.params:
            for bayesian_weights in bayesian_layers:
                mean_i = bayesian_weights[0]
                mean_list.append(mean_i)

        sample = distribution.sample()
        samples_list = utils.unflatten_like(sample, mean_list)
        i = 0
        for k in range(len(self.base_model.layers)):
            base_layer = self.base_model.layers[k]
            layer_parameters = base_layer.get_weights()
            w_list = []
            for _ in range(len(layer_parameters)):
                w_list.append(samples_list[i])
                i += 1
            base_layer.set_weights(w_list)

        return self.base_model.predict_on_batch(input)
            
    def step(self):

        input, label = next(self.dataloader)
        self.base_model.fit(input, label, epochs = 1, batch_size = 1)
  

        for bayesian_layers, base_layer in zip(self.params, self.base_model.layers):
            layer_parameters = base_layer.get_weights()
            for bayesian_weights, layer_weights  in zip(bayesian_layers, layer_parameters):
                if self.n_models % self.hyperparameters.frequency == 0:
                    mean = bayesian_weights[0]
                    sq_mean = bayesian_weights[1]
                    mean = (mean * self.n_models + layer_weights) / (self.n_models + 1.0)
                    
                    sq_mean = (sq_mean * self.n_models  + layer_weights ** 2) / (self.n_models + 1.0)

                    bayesian_weights[0] = mean;
                    bayesian_weights[1] = sq_mean
                    new_params = tf.transpose(tf.reshape((layer_weights - mean), [-1, 1]))
                    deviation_matrix = bayesian_weights[2]

                    if(deviation_matrix.shape[0] == self.hyperparameters.k):
                        bayesian_weights[2] = tf.concat(
                            (deviation_matrix[:self.hyperparameters.k - 1,:], new_params), axis = 0)
                    else:
                        bayesian_weights[2] = tf.concat(
                            (deviation_matrix, new_params), axis = 0)

        self.n_models += 1
