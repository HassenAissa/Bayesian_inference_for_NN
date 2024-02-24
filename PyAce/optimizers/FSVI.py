# Function space variational inference
from PyAce.optimizers import Optimizer
from PyAce.distributions import Distribution
import tensorflow as tf
import tensorflow_probability as tfp

class FSVI(Optimizer):
    def __init__(self):
        super().__init__()
        self._data_iterator = None
        self._dataloader = None
        self._layers_dtbn = []

    def predict_region(self, predict_data: tf.Tensor):
        x_mean = tf.math.reduce_mean(predict_data, axis=0)
        x_std = tf.math.reduce_std()
        self._predict_dtbn = tfp.distributions.Normal(x_mean, x_std)

    def sample_training(self):
        sample,label = next(self._data_iterator, (None,None))
        # if the iterator reaches the end of the dataset, reinitialise the iterator
        if sample is None:
            self._data_iterator = iter(self._dataloader)
            sample, label = next(self._data_iterator, (None, None))
        return sample
    
    def sample_predicts(self, pnum):
        pred_samples = []
        for n in range(pnum):
            s = self._predict_dtbn.sample()
            pred_samples.append(s)
        xs = tf.convert_to_tensor(pred_samples)
        return tf.reshape(xs, shape=(1,-1)) 
    
    def vi_noise(self):
        layers_weights = self._vi_model.layers.get_weights()
        for layer_w in layers_weights:
            w_mean = tf.fill(layer_w.shape, 0)
            w_std = tf.fill(layer_w.shape, self._vi_std)
            dtbn = tfp.distributions.Normal(w_mean, w_std) 
            self._layers_dtbn.append(dtbn)

    def vi_fwd_noise(self, t_samples, p_samples):
        layers_weights = self._vi_model.layers.get_weights()
        vi_weights = []
        for l in range(len(self._layers_dtbn)):
            layer_noise = self._layers_dtbn[l].sample()
            vi_w = tf.math.add(layers_weights[l], layer_noise)
            vi_weights.append(vi_w)
        self._vi_model.set_weights(vi_weights)

        t_funcs = self._vi_model(t_samples, training=True)
        p_funcs = self._vi_model(p_samples, training=True)
        return t_funcs, p_funcs
    
    def step(self, save_document_path=None):
        
        t_samples = self.sample_training()
        pnum = self._hyperparameters.pred_num
        p_samples = self.sample_predicts(pnum)
        for i in range(self._k_vifunc):
            t_funcs, p_funcs = self.vi_fwd_noise(t_samples, p_samples)
        
        
        return super().step(save_document_path)


    def compile_extra_components(self, **kwargs):
        self._vi_model = tf.keras.models.model_from_json(self._model_config)
        self._noise_std = self._hyperparameters.noise_std
        self._vi_std = self._hyperparameters.vi_std
        self._k_vifunc = self._hyperparameters.k_vifunc
        self._prior = kwargs["prior"]
        self._lr = self._hyperparameters.lr
        self._alpha = self._hyperparameters.alpha
        self.dataset_setup()
        self._priors_list = self._prior.get_model_priors(self._base_model)

