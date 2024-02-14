from PyAce.optimizers import HyperParametersSelector
from PyAce.distributions import GaussianPrior
from PyAce.optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Visualisation
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def runner():
    # x,y = sklearn.datasets.make_moons(n_samples=2000)
    # dataset = Dataset(
    #     tf.data.Dataset.from_tensor_slices((x, y)),
    #     tf.keras.losses.SparseCategoricalCrossentropy(),
    #     "Classification"    
    # )
    dataset = Dataset(r"C:\Users\hasse\Documents\Hassen\SEGP\Datasets\Boston.csv",tf.keras.losses.MeanSquaredError(), "Regression", feature_normalisation=True)

    base_model = tf.keras.Sequential()
    base_model.add(layers.Dense(10, activation='relu', input_shape=(13,)))
    base_model.add(tf.keras.layers.Dense(1, activation='linear'))

    # instantiate your optimizer
    optimizer = BBB
    prior = GaussianPrior(.0,0.0)


    # prior = GuassianPrior(0.0,2.01)
    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    hyperparameters_selector = HyperParametersSelector(dataset, lr = (1e-4,1e-2), alpha = (0.0,0.5))
    hyperparameters_selector.cross_validation(base_model.to_json(), optimizer, prior = prior)



