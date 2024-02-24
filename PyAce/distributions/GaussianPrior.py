import tensorflow as tf
import tensorflow_probability as tfp


class GaussianPrior:
    """
    class representing a Prioir following a normal distribution of a given mean and standard deviation
    """
    def __init__(self, mean, std_dev):
        if(type(mean)) != type(std_dev):
            raise Exception("mean and std dev must have the same type")
        self._mean = mean
        self._std_dev = std_dev

    def _get_priors_from_int_or_float(self, model):
        """
        creates a list of priors following the model trainable variables for a prior created by integer or floats
        Args:
            model (tf.keras.model): The model for which we create the priors

        Returns:
            list: list of priors following the model trainable variables
        """
        priors_list = []
        for layer_idx in range(len(model.layers)):
            if len(model.layers[layer_idx].trainable_variables) != 0:
                layer_distribs = []
                for w in model.layers[layer_idx].trainable_variables:
                    layer_distribs.append(tfp.distributions.Normal(self._mean * tf.ones(w.shape),
                                                                   self._std_dev * tf.ones(w.shape)))
                priors_list.append(layer_distribs)
            else:
                priors_list.append(None)
        return priors_list
    

    def _get_priors_from_list(self, model):
        """
        creates a list of priors following the model trainable variables for a prior created by a list of means and std_dev
        Args:
            model (tf.keras.model): The model for which we create the priors

        Returns:
            list: list of priors following the model trainable variables
        """
        priors_list = []
        for layer_idx in range(len(model.layers)):
            if len(model.layers[layer_idx].trainable_variables) != 0:
                layer_distribs = []
                for w in model.layers[layer_idx].trainable_variables:
                    layer_distribs.append(tfp.distributions.Normal(self._mean[layer_idx]*tf.ones(w.shape), self._std_dev[layer_idx]*tf.ones(w.shape)))
                priors_list.append(layer_distribs)
            else:
                priors_list.append(None)

        return priors_list
    
    def _get_priors_from_tensor(self, model):
        """
        creates a list of priors following the model trainable variables for a prior created by a tensor
        Args:
            model (tf.keras.model): The model for which we create the priors

        Returns:
            list: list of priors following the model trainable variables
        """
        priors_list = []        
        for layer_idx in range(len(model.layers)):
            layer = model.layers[layer_idx]

            if len(layer.trainable_variables) != 0:
                for i in range(len(layer.trainable_variables)):
                    if layer.trainable_variables[i].shape != self._mean[layer_idx][i].shape:
                        raise Exception(
                            "the shape of the mean tensor does not correspond to the shape of the model layer. Given shape: " \
                            + str(self._mean[layer_idx][i].shape) + ". Expected shape: " + str(
                                layer.trainable_variables[i].shape))
                    if layer.trainable_variables[i].shape != self._std_dev[layer_idx][i].shape:
                        raise Exception(
                            "the shape of the standard deviation tensor does not correspond to the shape of the model layer. Given shape: " \
                            + str(self._mean[layer_idx][i].shape) + ". Expected shape: " + str(
                                layer.trainable_variables[i].shape))
                priors_list.append([tfp.distributions.Normal(mean, std_dev) for mean, std_dev in zip(self._mean[layer_idx], self._std_dev[layer_idx])])
            else:
                priors_list.append(None)
    
    def get_model_priors(self, model):
        """
        creates a list of priors following the model trainable variables shape. 
        Each trainable weight gets a prior distribution assigned by the mean 
        and the std_dev of the class

        Args:
            model (tf.keras.model): the model to follow

        Raises:
            Exception: if the mean or rho are not int, float, list or tensor

        Returns:
            list: list of priors following the model trainable variables
        """
        if isinstance(self._mean, int) or isinstance(self._mean, float):
            return self._get_priors_from_int_or_float(model)
        if isinstance(self._mean, list) and (all(isinstance(m, int) for m in self._mean) or all(isinstance(m, float) for m in self._mean)):
            return self._get_priors_from_list(model)
        if isinstance(self._mean, list) and all(isinstance(l, list) for l in self._mean):
            return self._get_priors_from_tensor(model)
        raise Exception("mean and standard deviation should be an int, a float, a list or a tensor")

    
