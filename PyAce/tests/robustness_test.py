import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers.hyperparameters import HyperParameters
from PyAce.optimizers import BBB, SWAG, HMC, SGLD, VADAM, ADAM
from PyAce.distributions import GaussianPrior
from PyAce.visualisations import Robustness, Metrics
import tensorflow_datasets as tfds
import sys

def BBB_test(dataset, base_model):
    hyperparams = HyperParameters(lr=1e-1, alpha=1/7000, pi=0.7, batch_size=7000)
    # instantiate your optimizer
    optimizer = BBB()
    prior1 = GaussianPrior(0.0, -1.0)
    prior2 = GaussianPrior(0.0, -5.0)
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, prior=prior1, prior2=prior2)

    optimizer.train(300)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def SWAG_test(dataset, base_model):
    hyperparams = HyperParameters(lr = 1e-2, k = 500, scale = 1, frequency = 1, batch_size = 32)
    # instantiate your optimizer
    optimizer = SWAG()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(3000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def HMC_test(dataset, base_model):
    hyperparams = HyperParameters(epsilon=0.0005,L=300,m=2)
    # instantiate your optimizer
    prior = GaussianPrior(
        0.0, 1.0
    )
    optimizer = HMC()
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior=prior, nb_burn_epochs=20)
    optimizer.train(60)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def lr_(step):
    return 0.05*(250.0+step)**(-0.6)

def SGLD_test(dataset, base_model):
    hyperparams = HyperParameters(lr=lr_, frequency=1, batch_size=1)

    optimizer = SGLD()
    optimizer.compile(hyperparams, base_model.to_json(), dataset)
    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def VADAM_test(dataset, base_model):
    hyperparams = HyperParameters(
        lr=0.01,
        scale=1,
        frequency=1,
        num_data=506,  # BOSTON
        lam=25,
        beta_1=0.9, 
        beta_2=1-sys.float_info.epsilon, # as big as it gets, 0.9999999999999998
        batch_size=128,
    )

    optimizer = VADAM()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def ADAM_test(dataset, base_model):
    hyperparams = HyperParameters(
        lr=0.01,
        beta_1=0.9, 
        beta_2=1-sys.float_info.epsilon, # as big as it gets, 0.9999999999999998
        batch_size=128,
    )

    optimizer = ADAM()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def runner_regression():
    dataset = Dataset(
        "/Users/rosa/Downloads/Datasets/WineQT.csv",
        tf.keras.losses.MeanSquaredError,
        "Regression", feature_normalisation=True,
    )

    base_model = tf.keras.models.Sequential()
    base_model.add(layers.Dense(30, activation='relu', input_shape=(11,)))
    base_model.add(layers.Dense(1, activation='linear'))

    bayesian_model: BayesianModel = SWAG_test(dataset, base_model)
    #save_path = "/Users/rosa/Downloads/Testing SWAG_Wine"
    #bayesian_model.store(save_path)
    #bayesian_model = BayesianModel.load(save_path)
    print("Starting robustness analysis")
    # path = "PyAce/tests"
    robustness_builder = Robustness.Robustness(bayesian_model, dataset)
    
    robustness_builder.mean_corruption_error()
    robustness_builder.mean_corruption_error(relative=True)
    #robustness_builder.corruption_robustness_by_severity()
    robustness_builder.adversarial_robustness()
    
def runner_classification():
    dataset = Dataset(
        "mnist",
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification"    
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
    
    #bayesian_model: BayesianModel = SWAG_test(dataset, base_model)
    #save_path = "Models/HMC/Boston"
    #bayesian_model.store(save_path)
    path = "/Users/rosa/Downloads/Testing SWAG"
    bayesian_model = BayesianModel.load(path)
    print("Starting robustness analysis")
    
    robustness_builder = Robustness.Robustness(bayesian_model, dataset)
    
    # corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    #                         "gaussian_blur", "contrast", "brightness", "saturate","pixelate"]
    # for c in corruptions:
    #     robustness_builder.corruption_error(corruption=c)
    
    # robustness_builder.mean_corruption_error()
    # robustness_builder.mean_corruption_error(relative=True)
    robustness_builder.adversarial_robustness()
    
#runner_regression()
runner_classification()