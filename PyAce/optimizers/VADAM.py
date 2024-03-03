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
            # print("\n Loss after epoch %s: "%(self._epoch_num), self._running_loss / self._seen_batches)
            self._data_iterator = iter(self._dataloader)
            self._seen_batches = 1
            self._running_loss = 0
            self._epoch_num += 1
            sample, label = next(self._data_iterator, (None, None))
        
        # Start by adding noise to the parameter 
        for layer, layer_idx in zip(self._base_model.layers, range(len(self._base_model.layers))):
            for sublayer, sublayer_idx in zip(layer.trainable_variables, range(len(layer.trainable_variables))):
                eps = tf.random.normal(shape=sublayer.shape, mean = 0.0, stddev=1.0)
                sigma = 1/tf.sqrt(self._num_data * (self._v[layer_idx][sublayer_idx] + self._lam))
                #print("peturb norm: ", tf.norm(sigma))
                sublayer.assign_add(eps * sigma)
            
        with tf.GradientTape(persistent=True) as tape:
            predictions = self._base_model(sample, training = True)
            loss = self._dataset.loss()(label, predictions)
            self._running_loss += loss
            # save the loss if the path is specified
            if save_document_path != None:
                with open(save_document_path, "a") as losses_file:
                    losses_file.write(str(loss.numpy()))
        

        for layer, layer_idx in zip(self._base_model.layers, range(len(self._base_model.layers))):
            for sublayer, sublayer_idx in zip(layer.trainable_variables, range(len(layer.trainable_variables))):
                sublayer_grad = tape.gradient(loss, sublayer)
                if sublayer_grad is not None:
                    self._m[layer_idx][sublayer_idx] = self._beta_1 * self._m[layer_idx][sublayer_idx]
                    self._m[layer_idx][sublayer_idx] += (1 - self._beta_1) * (sublayer_grad + (self._lam * sublayer/self._num_data))
                    self._v[layer_idx][sublayer_idx] = self._beta_2 * self._v[layer_idx][sublayer_idx] + (1 - self._beta_2) * sublayer_grad**2
                    self._m_hat[layer_idx][sublayer_idx] = self._m[layer_idx][sublayer_idx] /(1 - (self._beta_1**self._epoch_num))
                    self._v_hat[layer_idx][sublayer_idx] = self._v[layer_idx][sublayer_idx] /(1 - (self._beta_2**self._epoch_num))
                    sublayer.assign_sub(self._lr * self._m_hat[layer_idx][sublayer_idx]
                                        /(tf.sqrt(self._v_hat[layer_idx][sublayer_idx])
                                           + self._lam/self._num_data))  # assign_sub for SGD update

        
        return self._running_loss / self._seen_batches

    def _init_adam_arrays(self):
        """
        initialise the mean, second moment (sq_mean), deviation and trainable weights lists
        """
        self._m = []
        self._m_hat = []
        self._v = []
        self._v_hat = []
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            if len(layer.trainable_variables) != 0:
                m_list = []
                m_hat_list = []
                v_list = []
                v_hat_list = []
                for var in layer.trainable_variables:
                    m_list.append(tf.zeros(var.shape, dtype=tf.float32))
                    m_hat_list.append(tf.zeros(var.shape, dtype=tf.float32))
                    v_list.append(tf.zeros(var.shape, dtype=tf.float32))
                    v_hat_list.append(tf.zeros(var.shape, dtype=tf.float32))
                self._m.append(m_list)
                self._m_hat.append(m_hat_list)
                self._v.append(v_list)
                self._v_hat.append(v_hat_list)
            else:
                self._m.append(None)
                self._m_hat.append(None)
                self._v.append(None)
                self._v_hat.append(None)
                
    def compile_extra_components(self, **kwargs):
        """
            compiles components of subclasses
            Args:
                starting_model: this is the starting model for the inference method. It could be a pretrained model.
        """
        self._frequency = self._hyperparameters.frequency
        self._lr = self._hyperparameters.lr
        self._scale = self._hyperparameters.scale
        self._batch_size = int(self._hyperparameters.batch_size)
        self._base_model = tf.keras.models.clone_model(kwargs["starting_model"])
        self._base_model.set_weights(kwargs["starting_model"].get_weights())
        self._dataloader = (self._dataset.training_dataset()
                            .shuffle(self._dataset.training_dataset().cardinality())
                            .batch(self._batch_size))
        self._init_adam_arrays()
        self._data_iterator = iter(self._dataloader)
        self._n = 0
        self._lam = getattr(self._hyperparameters, 'lam', None) if hasattr(self._hyperparameters, 'lam') else self._lam
        self._num_data = self._hyperparameters.num_data
        self._beta_1 = self._hyperparameters.beta_1
        self._beta_2 = self._hyperparameters.beta_2

        
    def result(self) -> BayesianModel:
        self._mean = []
        self._var = []
        # for x in self._v:
        #     print(x.shape)
        model = BayesianModel(self._model_config)

        for layer, layer_idx in zip(self._base_model.layers, range(len(self._base_model.layers))):
            if len(layer.trainable_variables) != 0:
                mean = [tf.reshape(i, (-1, 1)) for i in layer.trainable_variables]
                mean = tf.reshape(tf.concat(mean, 0), (-1, 1))
                var = [tf.reshape(i, (-1, 1)) for i in self._v[layer_idx]]
                var = tf.reshape(tf.concat(var, 0), (-1, 1))

                tf_dist = tfp.distributions.Normal(loc=tf.reshape(mean, (-1,)), scale=tf.reshape(var, (-1,)))
                tf_dist = TensorflowProbabilityDistribution(
                    tf_dist
                )
                model.apply_distribution(tf_dist, layer_idx, layer_idx)
        return model
    def update_parameters_step(self):
        pass
