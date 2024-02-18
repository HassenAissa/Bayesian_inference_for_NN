from PyAce.optimizers import BBB
from PyAce.distributions import GaussianPrior
import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Visualisation
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

    hyperparams = HyperParameters(lr=1e-3, alpha = 0.00)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-10.0)

    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior)

    loss_save_file = r"package/src/visualisations/loss_save_file"
    optimizer.train(100, loss_save_file)



    bayesian_model: BayesianModel = optimizer.result()
    # store_path = r"..."
    # bayesian_model.store(store_path)
    # bayesian_model: BayesianModel= BayesianModel.load(store_path)

    analytics_builder = Visualisation(bayesian_model)
    robustness_builder = Robustness(bayesian_model)

    print("Starting performence analysis")
    analytics_builder.visualise(dataset, 100, loss_save_file)
    #print("Starting robustness analysis")
    #robustness_builder.c_robustness(dataset, 100)

