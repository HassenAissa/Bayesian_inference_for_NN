from Pyesian.optimizers import SGLD
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from Pyesian.datasets import Dataset
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import HMC
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.optimizers import SWAG
from Pyesian.visualisations import Metrics
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def BBB_test(dataset, base_model):
    hyperparams = HyperParameters(lr=5*1e-1, alpha = 1/32, pi = 1, batch_size = 32)
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
    optimizer.compile(hyperparams, base_model.to_json(), dataset, starting_model = base_model, batch_size = 128)
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
    tests = [BBB_test, SWAG_test, SGLD_test]
    names = ["Testing BBB", "Testing SWAG", "Testing SGLD"]
    for test, name in zip(tests, names):
        print(name)
        bayesian_model = test(dataset, base_model)
        # store_path = r"..."
        # bayesian_model.store(store_path)
        # bayesian_model: BayesianModel= BayesianModel.load(store_path)
        analytics_builder = Metrics(bayesian_model, dataset)
        analytics_builder.summary(30)


# runner()