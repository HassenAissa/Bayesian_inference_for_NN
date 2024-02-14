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
from PyAce.visualisations import Visualisation
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def BBB_test(dataset, base_model):
    hyperparams = HyperParameters(lr=5*1e-1, alpha = 0.0001, pi = 1)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-2.0)
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior)

    optimizer.train(50)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def SWAG_test(dataset, base_model):
    hyperparams = HyperParameters(lr=1e-2, k=10, frequency=1, scale=1)
    # instantiate your optimizer
    optimizer = SWAG()
    optimizer.compile(hyperparams, base_model.to_json(), dataset, starting_model = base_model)
    optimizer.train(100)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def HMC_test(dataset, base_model):
    hyperparams = HyperParameters(epsilon=1e-3, L=100, m=2)
    # instantiate your optimizer
    prior = GaussianPrior(
        0.0, 1.0
    )
    optimizer = HMC()
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior=prior)
    optimizer.train(100, nb_burn_epoch=10)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def lr_(step):
    return 0.1 - step * 0.001

def SGLD_test(dataset, base_model):
    hyperparams = HyperParameters(lr=lr_, k=10, frequency=1, batch_size=5)

    optimizer = SGLD()
    optimizer.compile(hyperparams, base_model.to_json(), dataset)
    optimizer.train(100)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model



def runner():
    x,y = sklearn.datasets.make_moons(n_samples=2000)
    dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.SparseCategoricalCrossentropy(),
    "Classification"    
    )
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    base_model = tf.keras.Sequential()

    base_model.add(layers.Dense(50, activation='relu', input_shape=(2,)))
    base_model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))
    tests = [HMC_test, BBB_test, SWAG_test, SGLD_test]
    names = ["Testing HMC","Testing BBB", "Testing SWAG", "Testing SGLD"]
    for test, name in zip(tests, names):
        print(name)
        bayesian_model = test(dataset, base_model)
        # store_path = r"..."
        # bayesian_model.store(store_path)
        # bayesian_model: BayesianModel= BayesianModel.load(store_path)
        analytics_builder = Visualisation(bayesian_model)
        analytics_builder.visualise(dataset, 2)


