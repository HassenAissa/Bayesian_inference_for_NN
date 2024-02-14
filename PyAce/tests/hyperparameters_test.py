from optimizers.HyperparametersSelector import HyperParametersSelector
from distributions.GaussianPrior import GaussianPrior
from optimizers.BBB import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from datasets.Dataset import Dataset
from nn.BayesianModel import BayesianModel
from optimizers.HyperParameters import HyperParameters
from optimizers.SWAG import SWAG
from visualisations.Visualisation import Visualisation
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
    dataset = Dataset(r"C:\Users\hasse\Documents\Hassen\SEGP\Datasets\Boston.csv",tf.keras.losses.MeanSquaredError(), "Regression", normalise=True)

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



