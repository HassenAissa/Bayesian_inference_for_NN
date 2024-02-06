import tensorflow as tf
import tensorflow_probability as tfp


class GuassianPrior:
    def __init__(self, mean, std_dev):
        if(type(mean)) != type(std_dev):
            raise Exception("mean and std dev must have the same type")
        self._mean = mean
        self._std_dev = std_dev

    def _get_priors_from_int(self, model):
        priors_list = []
        for layer_idx in range(len(model.layers)):
            layer = model.layers[layer_idx]
            size = 0
            for w in layer.get_weights():
                size += tf.size(w).numpy()
            if size != 0:
                priors_list.append(tfp.distributions.Normal(
                    tf.fill((size,1), self._mean),
                    tf.fill((size,1), self._std_dev)
                ))
        return priors_list
    

    def _get_priors_from_list(self, model):
        priors_list = []
        mean_index = 0
        for layer_idx in range(len(model.layers)):
            layer = model.layers[layer_idx]
            size = 0
            for w in layer.get_weights():
                size += tf.size(w).numpy()
            if size != 0:
                priors_list.append(tfp.distributions.Normal(
                    tf.fill((size,1), self._mean[mean_index]),
                    tf.fill((size,1), self._std_dev[mean_index])
                ))
                mean_index += 1
        return priors_list
    
    def _get_priors_from_tensor(self, model):
        priors_list = []        
        for layer_idx in range(len(model.layers)):
            layer = model.layers[layer_idx]

            for i in range(len(layer.get_weights())):
                if layer.get_weights()[i].shape != self._mean[layer_idx][i].shape:
                    raise Exception("the shape of the mean tensor does not correspond to the shape of the model layer. Given shape: "\
                                     + str(self._mean[layer_idx][i].shape) + ". Expected shape: " + str(layer.get_weights()[i].shape))
                if layer.get_weights()[i].shape != self._std_dev[layer_idx][i].shape:
                    raise Exception("the shape of the standard deviation tensor does not correspond to the shape of the model layer. Given shape: "\
                                     + str(self._mean[layer_idx][i].shape) + ". Expected shape: " + str(layer.get_weights()[i].shape))
            mean = [tf.reshape(i, (-1, 1)) for i in self._mean[layer_idx]]
            mean = tf.reshape(tf.concat(mean, 0), (-1, 1))
            std_dev = [tf.reshape(i, (-1, 1)) for i in self._std_dev[layer_idx]]
            std_dev = tf.reshape(tf.concat(std_dev, 0), (-1, 1))
            priors_list.append(tfp.distributions.Normal(mean, std_dev))
        return priors_list
    
    def get_model_priors(self, model):
        if isinstance(self._mean, int) or isinstance(self._mean, float):
            return self._get_priors_from_int(model)
        if isinstance(self._mean, list) and (all(isinstance(m, int) for m in self._mean) or all(isinstance(m, float) for m in self._mean)):
            return self._get_priors_from_list(model)
        if isinstance(self._mean, list) and all(isinstance(l, list) for l in self._mean):
            return self._get_priors_from_tensor(model)
        raise Exception("mean and standard deviation should be an int, a float, a list or a tensor")

    
