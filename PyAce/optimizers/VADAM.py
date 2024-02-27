from math import sqrt
import os

from PyAce.distributions import MultivariateNormalDiagPlusLowRank
from PyAce.distributions.tf import TensorflowProbabilityDistribution
from PyAce.nn import BayesianModel
from . import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import copy

# Implimentation of https://arxiv.org/pdf/1806.04854.pdf


class VADAM(Optimizer):

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
        self._running_loss = 0
        self._seen_batches = 0
        self._total_batches = 0
        self._epoch_num = 1
        self._lam = 0.5

    def step(self, save_document_path = None):
        # get the sample and the label
        sample,label = next(self._data_iterator, (None,None))
        self._seen_batches += 1
        self._total_batches += 1
        # if the iterator reaches the end of the dataset, reinitialise the iterator
        if sample is None:
            print("\n Loss after epoch %s: "%(self._epoch_num), self._running_loss / self._seen_batches)
            self._data_iterator = iter(self._dataloader)
            self._seen_batches = 1
            self._running_loss = 0
            self._epoch_num += 1
            sample, label = next(self._data_iterator, (None, None))
        
        # Start by adding noise to the parameter 
        ind = 0
        for var in self._base_model.trainable_variables:
            eps = tf.random.normal(shape=var.shape, mean = 0.0, stddev=1.0)
            sigma = 1/tf.sqrt(self._num_data * (self._v[ind] + self._lam))
            #print("peturb norm: ", tf.norm(sigma))
            var.assign_add(eps * sigma)
            ind += 1
            
        with tf.GradientTape(persistent=True) as tape:
            predictions = self._base_model(sample, training = True)
            loss = self._dataset.loss()(label, predictions)
            self._running_loss += loss
            # save the loss if the path is specified
            if save_document_path != None:
                with open(save_document_path, "a") as losses_file:
                    losses_file.write(str(loss.numpy()))
        
        var_grad = tape.gradient(loss, self._base_model.trainable_variables)
        it_val = 0
        for var, grad in zip(self._base_model.trainable_variables, var_grad):
            if grad is not None:
                self._m[it_val] = self._hyperparameters.beta_1 * self._m[it_val] 
                self._m[it_val] += (1 - self._hyperparameters.beta_1) * (grad + (self._lam * var/self._num_data))
                self._v[it_val] = self._hyperparameters.beta_2 * self._v[it_val] + (1 - self._hyperparameters.beta_2) * grad**2
                self._m_hat[it_val] = self._m[it_val] /(1 - (self._hyperparameters.beta_1**self._epoch_num))
                self._v_hat[it_val] = self._v[it_val] /(1 - (self._hyperparameters.beta_2**self._epoch_num))
                var.assign_sub(self._lr * self._m_hat[it_val]/(tf.sqrt(self._v_hat[it_val]) + self._lam/self._num_data))  # assign_sub for SGD update
                it_val += 1
        
        return self._running_loss / self._seen_batches

    def _init_adam_arrays(self):
        """
        initialise the mean, second moment (sq_mean), deviation and trainable weights lists
        """
        self._m = []
        self._m_hat = []
        self._v = []
        self._v_hat = []
        for var in self._base_model.trainable_variables: 
            self._m.append(tf.zeros(var.shape, dtype=tf.float32))
            self._m_hat.append(tf.zeros(var.shape, dtype=tf.float32))
            self._v.append(tf.zeros(var.shape, dtype=tf.float32))
            self._v_hat.append(tf.zeros(var.shape, dtype=tf.float32))
        for layer_idx in range(len(self._base_model.layers)):
            self._weight_layers_indices.append(layer_idx)
                
    def compile_extra_components(self, **kwargs):
        self._frequency = self._hyperparameters.frequency
        self._lr = self._hyperparameters.lr
        self._scale = self._hyperparameters.scale
        self._base_model = tf.keras.models.clone_model(kwargs["starting_model"])
        self._base_model.set_weights(kwargs["starting_model"].get_weights())
        self._dataloader = (self._dataset.training_dataset()
                            .shuffle(self._dataset.training_dataset().cardinality())
                            .batch(
                                self._hyperparameters.batch_size
                            ))
        self._init_adam_arrays()
        self._data_iterator = iter(self._dataloader)
        self._n = 0
        self._lam = getattr(self._hyperparameters, 'lam', None) if hasattr(self._hyperparameters, 'lam') else self._lam
        self._num_data = self._hyperparameters.num_data
        
    def result(self) -> BayesianModel:
        self._mean = []
        self._var = []
        for x in self._v:
            print(x.shape)
        model = BayesianModel(self._model_config)
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]; size = 0
            for w in layer.trainable_variables:
                if(size == 0):
                    init_val = tf.dtypes.cast(tf.reshape(w, (-1)), dtype=tf.float32)
                    init_var = tf.dtypes.cast(tf.reshape(self._v[2*layer_idx], (-1)), dtype=tf.float32) 
                else: 
                    init_val = tf.concat((init_val, tf.dtypes.cast(tf.reshape(w, (-1)), dtype=tf.float32)), axis=0)
                    init_var = tf.concat((init_var, tf.dtypes.cast(tf.reshape(self._v[2*layer_idx + 1], (-1)), dtype=tf.float32)), axis=0)
                size += tf.size(w).numpy()
            self._mean.append(tf.expand_dims(init_val, axis=-1))
            self._var.append(tf.expand_dims(init_var, axis=-1))
        idx = 0    
        for mean, var, idx in zip(self._mean, self._var, range(len(self._weight_layers_indices))):
            tf_dist = tfp.distributions.Normal(loc=tf.reshape(mean, (-1,)), scale=tf.reshape(var, (-1,)))
            #tf_dist = TensorflowProbabilityDistribution(tf_dist)
            start_idx = self._weight_layers_indices[idx]
            end_idx = len(self._base_model.layers) - 1
            if idx + 1 < len(self._weight_layers_indices):
                end_idx = self._weight_layers_indices[idx + 1]
            model.apply_distribution(tf_dist, start_idx, start_idx)
            idx+=1
        return model

    def update_parameters_step(self):
        pass