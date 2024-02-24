from PyAce.distributions import GaussianPrior
from PyAce.optimizers import BBB
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers.hyperparameters import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Metrics

def runner():
    x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
    y = 2*x+2
    dataset = tf.data.Dataset.from_tensor_slices((x, y))


    #train_dataset = Dataset(train_dataset, tf.keras.losses.MeanSquaredError())
    dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression", feature_normalisation= True)


    model = tf.keras.models.Sequential()
    model.add(layers.Dense(5, activation='tanh', input_shape=(1,))) 
    model.add(layers.Dense(1, activation='linear'))

    hyperparams = HyperParameters(lr=1e-1, alpha = 0.0)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(
        0.0,-10.0
        )
    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    optimizer.compile(hyperparams, model.to_json(), dataset, prior = prior)

    optimizer.train(100)



    bayesian_model: BayesianModel = optimizer.result()
    # store_path = r""
    # bayesian_model.store(store_path)
    # bayesian_model: BayesianModel= BayesianModel.load(store_path)

    analytics_builder = Metrics(bayesian_model)

    analytics_builder.visualise(dataset, 2000)


