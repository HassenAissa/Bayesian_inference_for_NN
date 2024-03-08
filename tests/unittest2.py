from PyAce.visualisations import Plotter
from PyAce.optimizers.SGLD import SGLD
from PyAce.distributions.GaussianPrior import GaussianPrior
from PyAce.optimizers import ADAM, BBB, BSAM, SGD
import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HMC
from PyAce.optimizers.VADAM import VADAM
from PyAce.optimizers.hyperparameters import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations.Metrics import Metrics
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp
import sys

def BBB_test(dataset, base_model):
    # hyperparams = HyperParameters(lr=1e-3, alpha = 1/128, pi = 0.4)

    # # instantiate your optimizer
    # optimizer = BBB()
    # prior1 = GaussianPrior(0.0,-5.0)
    # prior2 = GaussianPrior(0.0,0.01)
    hyperparams = HyperParameters(lr=5*1e-1, alpha=0, pi=0.5, batch_size=4096)

    # instantiate your optimizer
    optimizer = BBB()
    prior1 = GaussianPrior(0.0, -1.0)
    prior2 = GaussianPrior(0.0, -5.0)
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, prior=prior1, prior2=prior2)

    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def SWAG_test(dataset, base_model):
    hyperparams = HyperParameters(
        lr=1e-1, k=50, frequency=1,
        scale=1,
        batch_size=2000)
    # instantiate your optimizer
    optimizer = SWAG()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(3000)
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
    hyperparams = HyperParameters(lr_upper = 0.01, lr_lower = 0.003, lr_gamma = 0.99,  batch_size=10)

    optimizer = SGLD()
    optimizer.compile(hyperparams, base_model.to_json(), dataset)
    optimizer.train(10000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def VADAM_test(dataset, base_model):
    hyperparams = HyperParameters(
        lr=1e-2,
        num_data=45000,  # BOSTON
        lam=5000,
        beta_1=0.9, 
        beta_2=0.999, # as big as it gets, 0.9999999999999998
        batch_size=512,
    )

    optimizer = VADAM()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def ADAM_test(dataset, base_model):
    hyperparams = HyperParameters(
        lr=1e-3,
        beta_1=0.8, 
        beta_2=0.9999999, # as big as it gets, 0.9999999999999998
        batch_size=32,
    )


    optimizer = ADAM()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model

def SGD_test(dataset, base_model):
    hyperparams = HyperParameters(
        lr=1e-1,
        frequency = 1,
        batch_size=4096,
    )

    optimizer = SGD()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(8000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model



def BSAM_test(dataset, base_model):
    hyperparams = HyperParameters(lr=0.5, beta_1 = 0.9, beta_2 = 0.9999999, lam = 0.0, 
                                rho=0.00001, gam=0.1, batch_size= 256)

    optimizer = BSAM()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(2000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model




# x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
# y = 2*x+2
# dataset = tf.data.Dataset.from_tensor_slices((x, y))
# dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression", normalise= True)
# dataset = Dataset(r"C:\Users\hasse\Documents\Hassen\SEGP\Datasets\Boston.csv",
#                   tf.keras.losses.MeanSquaredError(),
#                   "Regression", feature_normalisation=True,
#                   )


# dataset = Dataset(
#     "C:/Users/hasse/Documents/Hassen/SEGP/Datasets/RetinaMNIST",
#     tf.keras.losses.SparseCategoricalCrossentropy(),
#     "Classification", 
#     load_images= True,
# )





def runner():


    # base_model.add(layers.Dense(30, activation='relu', input_shape=(13,)))


    base_model = models.Sequential()
    # base_model.add(layers.Conv2D(32, 3, activation='relu', ))
    # base_model.add(layers.MaxPooling2D(2))
    # base_model.add(layers.Conv2D(16, 3, activation='relu'))
    # base_model.add(layers.MaxPooling2D(2))
    # # base_model.add(layers.Conv2D(64, 3, activation='relu'))
    # # base_model.add(layers.MaxPooling2D(2))
    # # base_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # # base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    # # base_model.add(tf.keras.layers.MaxPooling2D(2))
    # base_model.add(layers.Conv2D(16, 3, activation='relu', input_shape=(32,32,3)))
    # base_model.add(layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))

    # # base_model.add(tf.keras.layers.Dense(120, activation='relu'))
    base_model.add(tf.keras.layers.Dense(120, activation='relu'))
    base_model.add(tf.keras.layers.Dense(50, activation='relu'))


    # base_model.add(tf.keras.layers.Dense(125, activation='relu'))

    # base_model.add(tf.keras.layers.Dense(128, activation='relu'))
    base_model.add(tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax))

    # dataset_name = "cifar10"
    # dataset = Dataset(
    #     dataset_name,
    #     tf.keras.losses.SparseCategoricalCrossentropy,
    #     "Classification",
    #     train_proportion=0.7,
    #     test_proportion=0.1,
    #     valid_proportion=0.2,
    #     feature_normalisation=True
    # )

    # base_model = tf.keras.models.Sequential()
    # # base_model.add(tf.keras.layers.Flatten(input_shape=(32,32,3)))
    # # # base_model.add(layers.Dense(800, activation='relu'))
    # # # base_model.add(layers.Dense(800, activation='relu'))
    # # base_model.add(layers.Dense(800, activation='relu'))
    # base_model.add(layers.Dense(30, activation='relu', input_shape=(16,)))
    # base_model.add(layers.Dense(30, activation='relu'))

    # base_model.add(layers.Dense(2, activation='linear'))

    dataset_name = "RetinaMnist"
    dataset = Dataset(
        "C:/Users/hasse/Documents/Hassen/SEGP/Datasets/"+dataset_name,
        tf.keras.losses.SparseCategoricalCrossentropy,
        "Classification",
        load_images=True,
        # feature_normalisation=True,
        # target_dim=2
        )

    tests = [
        # BSAM_test,
        ADAM_test,
        # SGD_test
        # SWAG_test,
        # BBB_test,
        # SGLD_test
    ]
    names = [
        # "Testing SWAG",
        # "Testing BBB",
        # "Testing VADAM",
        "Testing ADAM",
        # "Testing SGD",
        # "Testing BSAM" 
        # "Testing BBB",
        # "Testing SGLD"
    ]
    for test, name in zip(tests, names):
        print(name)
        bayesian_model = test(dataset, base_model)
        # store_path = r"./trained_models/"+dataset_name+"/"+name
        # bayesian_model.store(store_path)
        # bayesian_model: BayesianModel= BayesianModel.load(store_path)
        # print("finished storing")
        # print("start metrics")
        analytics_builder = Metrics(bayesian_model, dataset)
        analytics_builder.summary(n_boundaries=50, save_path=None)

runner()
