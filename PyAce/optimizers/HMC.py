import random

from PyAce.distributions import GaussianPrior
from PyAce.nn import BayesianModel
from . import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import random
import math
from PyAce.distributions import Sampled


class HMC(Optimizer):

    def __init__(self):
        super().__init__()
        self._training_dataset_cardinality = None
        self._burn_epochs = None
        self._training_dataset = None
        self._prior = None
        self._p = None
        self._frequency = None
        self._samples = None
        self._model = None
        self._epsilon = None
        self._L = None
        self._m = None
        self._total_runs = 0
        self._accepted_runs = 0
        self._current_loss = 0

    def compile_extra_components(self, **kwargs):
        self._m = self._hyperparameters.m
        self._L = self._hyperparameters.L
        self._epsilon = self._hyperparameters.epsilon
        self._model: tf.keras.models.Model = tf.keras.models.model_from_json(self._model_config)
        self._samples = []
        self._frequency = []
        self._p = []
        self._prior = kwargs["prior"].get_model_priors(self._model)
        self._training_dataset: tf.data.Dataset = self._dataset.training_dataset()
        self._training_dataset_cardinality = self._training_dataset.cardinality().numpy().item()
        for layer in self._model.layers:
            self._p.append([tf.Variable(tf.zeros(w.shape)) for w in layer.trainable_variables])

        for layer, distribs in zip(self._model.layers, self._prior):
            if len(layer.trainable_variables) > 0:
                for w, d in zip(layer.trainable_variables, distribs):
                    w.assign(d.mean())


    def step(self, save_document_path=None, sampling=True, burning = False):
        if len(self._frequency) == 0 and sampling:
            self._frequency.append(1)
            self._samples.append(self._snapshot_q())
        self._sample_kinetic_energy()
        current_k = self._kinetic_energy()
        current_u, current_loss = self._potential_energy()
        current_q = self._snapshot_q()
        self._step_p(self._epsilon / 2)
        for i in range(self._L):
            self._step_q(self._epsilon)
            if i != self._L:
                self._step_p(self._epsilon)
        self._step_p(self._epsilon / 2)
        new_k = self._kinetic_energy()
        new_u, new_loss = self._potential_energy()
        self._total_runs += 1
        if burning or random.random() < tf.math.exp(current_k + current_u - new_k - new_u).numpy().item():
            self._accepted_runs += 1
            if sampling:
                self._frequency.append(1)
                self._samples.append(self._snapshot_q())
            return new_loss
        else:
            # if rejected restore old layers variables
            for layer, current_layer in zip(self._model.layers, current_q):
                for w, q in zip(layer.trainable_variables, current_layer):
                    w.assign(q)
            if sampling:
                self._frequency[len(self._frequency) - 1] += 1
            return current_loss


    def train(self, nb_iterations: int, loss_save_document_path: str = None, model_save_frequency: int = None,
              model_save_path: str = None, nb_burn_epoch=10):
        self._accepted_runs = 0
        self._total_runs = 0
        for i in range(nb_burn_epoch):
            loss = self.step(sampling=False, burning=True)
            accept_rate = self._accepted_runs/self._total_runs
            self._print_progress((i+1) / nb_burn_epoch, suffix="HMC - Burning", loss=loss.numpy().item(), accept_rate=accept_rate, bar_length=20)
        print()
        self._accepted_runs = 0
        self._total_runs = 0
        self._frequency = []
        self._samples = []
        for i in range(nb_iterations):
            loss = self.step(sampling=True, burning=False)
            accept_rate = self._accepted_runs / self._total_runs
            self._print_progress((i+1)/nb_iterations, suffix="HMC - Sampling", loss=loss.numpy().item(), accept_rate=accept_rate, bar_length=20)
        print()



    def _step_p(self, step_size):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self._model.trainable_variables)
            U, _ = self._potential_energy()
        for layer, momentum_layer in zip(self._model.layers, self._p):
            for q, p in zip(layer.trainable_variables, momentum_layer):
                q_grad = tape.gradient(U, q)
                p.assign_sub(tf.multiply(q_grad, step_size))
        del tape # free the gradient resources

    def _step_q(self, step_size):
        for layer, momentum_layer in zip(self._model.layers, self._p):
            for q, p in zip(layer.trainable_variables, momentum_layer):
                q.assign_add(tf.multiply(p, step_size / self._m))

    def _snapshot_q(self):
        q = []
        for layer in self._model.layers:
            q.append([tf.identity(w) for w in layer.trainable_variables])
        return q

    def _potential_energy(self) -> (tf.Tensor, tf.Tensor):
        potential_energy = tf.constant([0.0])
        for layer, distribs in zip(self._model.layers, self._prior):
            if len(layer.trainable_variables) > 0:
                for w, d in zip(layer.trainable_variables, distribs):
                    potential_energy -= tf.math.reduce_sum(d.log_prob(w))
        X, y = next(iter(self._training_dataset.batch(self._training_dataset.cardinality())))
        predictions = self._model(X)
        # the loss is already the log likelihood of the data in this case
        loss = self._dataset.loss()(y, predictions)
        potential_energy += loss*self._training_dataset_cardinality
        return potential_energy, loss

    def _kinetic_energy(self):
        kinetic_energy = tf.constant([0.0])
        for layer in self._p:
            for var_p in layer:
                kinetic_energy += (1 / (2 * self._m)) * tf.math.reduce_sum(tf.square(var_p))
        return kinetic_energy

    def _sample_kinetic_energy(self):
        for layer in self._p:
            for w in layer:
                w.assign(tf.random.normal(w.shape, stddev=self._m, mean=0))

    def update_parameters_step(self):
        pass

    def result(self) -> BayesianModel:
        samples_unrolled = []
        for sample in self._samples:
            concat_unrolled = []
            for sample_layer in sample:
                for w in sample_layer:
                    concat_unrolled.append(tf.reshape(w, (-1,)))
            samples_unrolled.append(tf.concat(concat_unrolled, axis = 0))
        distribution = Sampled(samples_unrolled, self._frequency)
        posterior_model = BayesianModel(self._model_config)
        posterior_model.apply_distribution(distribution, 0, len(self._model.layers)-1)
        return posterior_model