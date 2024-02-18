from optimizers.SGLD import SGLD
from distributions.GaussianPrior import GaussianPrior
from optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from datasets import Dataset
from nn import BayesianModel
from optimizers import HMC
from optimizers import HyperParameters
from optimizers import SWAG
from visualisations.Visualisation import Visualisation
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def BBB_test(dataset, base_model):
    # hyperparams = HyperParameters(lr=1e-3, alpha = 1/128, pi = 0.4)

    # # instantiate your optimizer
    # optimizer = BBB()
    # prior1 = GaussianPrior(0.0,-5.0)  
    # prior2 = GaussianPrior(0.0,0.01)
    hyperparams = HyperParameters(lr=1e-3, alpha = 1/128, pi = 0.4)

    # instantiate your optimizer
    optimizer = BBB()
    prior1 = GaussianPrior(0.0,-5.0)  
    prior2 = GaussianPrior(0.0,0.01)
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior1, prior2 = prior2)

    optimizer.train(100)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def SWAG_test(dataset, base_model):
    hyperparams = HyperParameters(lr=1e-3, k=10, frequency=1, scale=1)
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
    return 0.01 - step * 0.0001

def SGLD_test(dataset, base_model):
    hyperparams = HyperParameters(lr=lr_, k=10, frequency=1, batch_size=5)

    optimizer = SGLD()
    optimizer.compile(hyperparams, base_model.to_json(), dataset)
    optimizer.train(100)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
y = 2*x+2
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression", normalise= True)
# dataset = Dataset(r"C:\Users\hasse\Documents\Hassen\SEGP\Datasets\Boston.csv",tf.keras.losses.MeanSquaredError(), "Regression", normalise=True)

def runner():
    base_model = tf.keras.models.Sequential()
    # base_model.add(layers.Dense(10, activation='relu', input_shape=(13,))) 
    base_model.add(layers.Dense(10, activation='relu', input_shape=(1,))) 
    # base_model.add(layers.Dense(30, activation='relu', input_shape=(13,))) 

    base_model.add(layers.Dense(1, activation='linear'))
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


