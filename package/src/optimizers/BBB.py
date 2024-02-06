import os
from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution
from src.nn.BayesianModel import BayesianModel
from src.optimizers.Optimizer import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp


class BBB(Optimizer):
    def __init__(self):
        super().__init__()
        self._data_iterator = None
        self._dataloader = None
        self._lr = None
        self._alpha = None
        self._base_model: tf.keras.Model = None
        self._priors_list = None
        self._weight_layers_indices = []
        self._posterior_mean_list = []
        self._posterior_std_dev_list = []
        self._priors_list = []
        self._layers_intervals = []

    def _guassian_likelihood(self,weights, mean, std_dev):
        """
        calculates the likelihood of the weights being from a guassian distribution of the given mean and standard deviation

        Args:
            weights (tf.Tensor): the weights to test
            mean (tf.Tensor): the mean
            std_dev (tf.Tensor): the standard deviation

        Returns:
            tf.Tensor: the likelihood
        """
        guassian_distribution = tfp.distributions.Normal(mean, tf.math.softplus(std_dev))
        return tf.reduce_mean(guassian_distribution.log_prob(weights)) #TODO: mena or sum, paper says sum but other sources say sum

    def _prior_guassian_likelihood(self):
        """
        calculates the guassian likelihood of the weights with respect to the prior
        Returns:
            tf.Tensor: the likelihood of the weights of being sampled from the prior
        """
        likelihood = 0
        mean_idx = 0
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            if len(layer.get_weights()) != 0:
                weights = [tf.reshape(i, (-1, 1)) for i in layer.get_weights()]
                weights = tf.reshape(tf.concat(weights, 0), (-1, 1))
                likelihood += self._guassian_likelihood(
                    weights, 
                    self._priors_list[mean_idx].mean(), 
                    self._priors_list[mean_idx].stddev()
                )
                mean_idx += 1

        return likelihood
    
    def _posterior_guassian_likelihood(self, mean_list, std_dev_list):
        """
        calculates the guassian likelihood of the weights with respect to the posterior
        Returns:
            tf.Tensor: the likelihood of the weights of being sampled from the posterior
        """
        likelihood = 0
        mean_idx = 0
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            if len(layer.get_weights()) != 0:
                weights = [tf.reshape(i, (-1, 1)) for i in layer.get_weights()]
                weights = tf.reshape(tf.concat(weights, 0), (-1, 1))
                likelihood += self._guassian_likelihood(
                    weights, 
                    mean_list[mean_idx], 
                    std_dev_list[mean_idx]
                )
                mean_idx += 1

        return likelihood

    def _cost_function(self, labels: tf.Tensor, predictions: tf.Tensor):
        """
        calculate the cost function as follows:
        cost = data_likelihood + alpha * kl_divergence

        Args:
            labels (tf.Tensor): the labels
            predictions (tf.Tensor): the predictions

        Returns:
            tf.Tensor: the cost
        """
        posterior_likelihood = self._posterior_guassian_likelihood(self._posterior_mean_list, self._posterior_std_dev_list)
        prior_likelihood = self._prior_guassian_likelihood()
        kl_divergence = tf.math.subtract(posterior_likelihood, prior_likelihood)
        data_likelihood = self._dataset.loss()(labels, predictions)
        kl_divergence = tf.multiply(kl_divergence, self._alpha)
        return tf.math.add(data_likelihood, kl_divergence)
    


    def step(self, save_document_path = None):
        #update the weights
        self._update_weights()

        # get sample and label
        sample,label = next(self._data_iterator, (None,None))
        # if the iterator reaches the end of the dataset, reinitialise the iterator
        if sample is None:
            self._data_iterator = iter(self._dataloader)
            sample, label = next(self._data_iterator, (None, None))

        with tf.GradientTape(persistent=True) as tape:
            # take the posterior distribution into account in the calculation of the gradients
            tape.watch(self._posterior_mean_list)
            tape.watch(self._posterior_std_dev_list)

            predictions = self._base_model(sample, training = True)
            likelihood = self._cost_function(
                label, 
                predictions
            )

            print("likelihood",likelihood)
        # get the weight, mean and standard deviation gradients
        weight_gradients = tape.gradient(likelihood, self._base_model.trainable_weights)
        mean_gradients = tape.gradient(likelihood, self._posterior_mean_list)
        std_dev_gradients = tape.gradient(likelihood, self._posterior_std_dev_list)

        # save the model loss if the path is specified
        if save_document_path != None:
            with open(save_document_path, "a") as losses_file:
                losses_file.write(str(likelihood.numpy())+"\n")

        new_posterior_mean_list = []
        new_posterior_std_dev_list = []
        trainable_layer_index = 0
        gradient_layer_index = 0
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            if len(layer.get_weights()) != 0:
                weight_gradient = []
                # go through weights and biases and merge their gradients into one vector
                for j in range(len(layer.get_weights())):
                    weight_gradient.append(tf.reshape(weight_gradients[gradient_layer_index], (-1, 1)))
                    gradient_layer_index += 1
                weight_gradient = tf.reshape(tf.concat(weight_gradient, 0), (-1, 1))

                # calculate the new mean of the posterior
                mean_gradient = mean_gradients[trainable_layer_index]+weight_gradient
                # calculate the new mean of the posterior
                new_posterior_mean_list.append(
                    self._posterior_mean_list[trainable_layer_index]-self._lr*mean_gradient
                )
                # calculate the std_dev gradient
                posterior_std_dev = self._posterior_std_dev_list[trainable_layer_index]
                noise = tfp.distributions.Normal(
                    tf.zeros(posterior_std_dev.shape),
                    tf.ones(posterior_std_dev.shape)
                ).sample()
                std_dev_grad = noise / (1 + tf.math.exp(-posterior_std_dev))
                std_dev_grad *= weight_gradient
                std_dev_grad += std_dev_gradients[trainable_layer_index]

                # calculate the new standard deviation of the posterior
                new_posterior_std_dev_list.append(
                    posterior_std_dev-self._lr*std_dev_grad
                )

                trainable_layer_index += 1
        #update the posteriors
        self._posterior_mean_list = new_posterior_mean_list
        self._posterior_std_dev_list = new_posterior_std_dev_list





    def _update_weights(self):
        """
        generate new weights following the posterior distributions
        """
        for interval_idx in range(len(self._layers_intervals)):
            #sample the new wights as a vector
            vector_weights = tfp.distributions.Normal(
                self._posterior_mean_list[interval_idx],
                tf.math.softplus(self._posterior_std_dev_list[interval_idx])
            ).sample()
            # if(interval_idx == 0):
            #     print("mean", self._posterior_mean_list[interval_idx][0])
            #     print("std_dev", tf.math.softplus(self._posterior_std_dev_list[interval_idx])[0])
            #     print(vector_weights[0])

            # reshape the vector and update the base model
            start = self._layers_intervals[interval_idx][0]
            end = self._layers_intervals[interval_idx][1]
            take_from = 0
            for layer_idx in range(start, end + 1):
                weights = []
                for w in self._base_model.layers[layer_idx].get_weights():
                    size = tf.size(w).numpy()
                    weights.append(tf.reshape(vector_weights[take_from:take_from + size], w.shape))
                    take_from += size
                self._base_model.layers[layer_idx].set_weights(weights)



    def compile_extra_components(self, **kwargs):
        self._base_model = tf.keras.models.clone_model(kwargs["starting_model"])
        self._base_model.set_weights(kwargs["starting_model"].get_weights())
        self._prior = kwargs["prior"]
        self._lr = self._hyperparameters.lr
        self._alpha = self._hyperparameters.alpha
        self._dataloader = (self._dataset.training_dataset()
                            .shuffle(self._dataset.training_dataset().cardinality())
                            .batch(self._dataset.training_dataset().cardinality()))
        self._data_iterator = iter(self._dataloader)
        self._priors_list = self._prior.get_model_priors(self._base_model)
        self._init_BBB_arrays()


    def _init_BBB_arrays(self):
        """
        initialises the posterior list, the correlated layers intervals and the trainable layer indices
        """
        trainable_layer_index = 0
        for layer_idx in range(len(self._base_model.layers)):
            layer = self._base_model.layers[layer_idx]
            size = 0
            for w in layer.get_weights():
                size += tf.size(w).numpy()
            if size != 0:
                self._layers_intervals.append([layer_idx, layer_idx])
                self._posterior_mean_list.append(self._priors_list[trainable_layer_index].mean())
                self._posterior_std_dev_list.append(self._priors_list[trainable_layer_index].stddev())
                self._weight_layers_indices.append(layer_idx)
                trainable_layer_index += 1

    def result(self) -> BayesianModel:
        model = BayesianModel(self._model_config)
        for mean, std_dev, idx in zip(self._posterior_mean_list, self._posterior_std_dev_list, range(len(self._weight_layers_indices))):
            tf.debugging.check_numerics(mean, "mean")
            tf.debugging.check_numerics(std_dev, "standard deviation")

            tf_dist = tfp.distributions.Normal(
                tf.reshape(mean, (-1,)),
                tf.math.softplus(tf.reshape(std_dev, (-1,)))
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

    def update_parameters_step(self):
        pass
