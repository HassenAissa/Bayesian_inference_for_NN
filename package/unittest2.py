from src.distributions.GaussianPrior import GaussianPrior
from src.optimizers.BBB import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HMC import HMC
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG
from src.visualisations.Visualisation import Visualisation
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def BBB_test(dataset, base_model):
    hyperparams = HyperParameters(lr=5*1e-1, alpha = 0.0001)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-2.0)
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior)

    optimizer.train(50)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def SWAG_test(dataset, base_model):
    hyperparams = HyperParameters(lr=1e-1, k=10, frequency=1, scale=1)
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

def SGLD_test(dataset, base_model):
    pass

x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
y = 2*x+2
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression", normalise= True)


initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
base_model = tf.keras.models.Sequential()
base_model.add(layers.Dense(5, activation='tanh', input_shape=(1,))) 
base_model.add(layers.Dense(1, activation='linear'))
tests = [HMC_test, BBB_test, SWAG_test]
names = ["Testing HMC", "Testing BBB", "Testing SWAG"]
for test, name in zip(tests, names):
    print(name)
    bayesian_model = test(dataset, base_model)
    # store_path = r"..."
    # bayesian_model.store(store_path)
    # bayesian_model: BayesianModel= BayesianModel.load(store_path)
    analytics_builder = Visualisation(bayesian_model)
    analytics_builder.visualise(dataset, 2)


