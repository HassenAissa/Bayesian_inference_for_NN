from PyAce.optimizers import SGLD
from PyAce.distributions import GaussianPrior
from PyAce.optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HMC
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Visualisation, Plotter
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

x,y = sklearn.datasets.make_moons(n_samples=2000)
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.BinaryCrossentropy(),
    "Classification"
)
'''
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
base_model = tf.keras.Sequential()
base_model.add(layers.Dense(50, activation='relu', input_shape=(2,)))
base_model.add(layers.Dense(30, activation='relu'))
base_model.add(layers.Dense(10, activation='relu'))
base_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))


hyperparams = HyperParameters(epsilon=0.003, L=30, m=2)
prior = GaussianPrior(
    0.0, 1.0
)
optimizer = HMC()
optimizer.compile(hyperparams, base_model.to_json(), dataset, prior=prior)
optimizer.train(100, nb_burn_epoch=10)
bayesian_model: BayesianModel = optimizer.result()

bayesian_model.store("testmodel")
'''
bayesian_model = BayesianModel.load("testmodel1sample")
plotter = Plotter(bayesian_model, dataset)
plotter.plot_decision_boundaries(granularity=0.0001, un_zoom_level=2)
plotter.plot_uncertainty_area(uncertainty_threshold=0.9, un_zoom_level=2)

# play and compate with one NN.

