from PyAce.optimizers import BBB
from PyAce.distributions import GaussianPrior
import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Metrics
from PyAce.visualisations import Robustness
import tensorflow_datasets as tfds

def runner():
    dataset = Dataset(
        "mnist",
        tf.keras.losses.SparseCategoricalCrossentropy(),
        "Classification",
    )

    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    base_model = tf.keras.Sequential()

    base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(120, activation='relu'))
    base_model.add(tf.keras.layers.Dense(84, activation='relu'))
    base_model.add(tf.keras.layers.Dense(62, activation=tf.keras.activations.softmax))

    hyperparams = HyperParameters(lr=1e-1, alpha = 1/4096, pi =1, batch_size = 4096)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-5.0)
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior)
    optimizer.train(1000)



    bayesian_model: BayesianModel = optimizer.result()
    analytics_builder = Metrics(bayesian_model, dataset)
    print("Starting performence analysis")
    analytics_builder.summary(100)



runner()