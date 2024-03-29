from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from Pyesian.datasets import Dataset
from Pyesian.nn import BayesianModel
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.optimizers import SWAG
from Pyesian.visualisations import Metrics, Plotter
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp


def runner():

    x,y = sklearn.datasets.make_moons(n_samples=2000)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy(),
        "Classification"    
    )
    print(dataset.input_shape())

    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    base_model = tf.keras.Sequential()

    base_model.add(layers.Dense(50, activation='tanh', input_shape=(2,)))
    base_model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

    # hyperparams = HyperParameters(lr=1e-3, k=50, frequency=1, scale=1)
    # # instantiate your optimizer
    # optimizer = SWAG()¨
    # # compile the optimizer with your data
    # # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    # optimizer.compile(hyperparams, base_model.get_config(), dataset, starting_model=base_model)


    hyperparams = HyperParameters(lr=5, alpha = 0.00, pi = 1, batch_size = 100)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-2.0)


    # prior = GuassianPrior(0.0,2.01)
    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior)



    optimizer.train(50)



    bayesian_model: BayesianModel = optimizer.result()
    # store_path = r"..."
    # bayesian_model.store(store_path)
    # bayesian_model: BayesianModel= BayesianModel.load(store_path)

    analytics_builder = Metrics(bayesian_model, dataset)
    analytics_builder.classification_uncertainty()
    analytics_builder.summary(35)
    plotter = Plotter(bayesian_model, dataset)
    plotter.entropy()

# runner()