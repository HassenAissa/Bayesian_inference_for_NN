from PyAce.optimizers import BBB
from PyAce.distributions import GaussianPrior
import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers.hyperparameters import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Metrics, Plotter
from PyAce.visualisations import Robustness
import tensorflow_datasets as tfds

def runner():
    dataset = Dataset(
        "cifar10",
        tf.keras.losses.SparseCategoricalCrossentropy(),
        "Classification",
        train_proportion=0.8,
        feature_normalisation=True
    )

    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    base_model = tf.keras.Sequential()

    base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32, 32, 3)))
    # base_model.add(tf.keras.layers.MaxPooling2D(2))
    # base_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    # base_model.add(tf.keras.layers.MaxPooling2D(2))
    # base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    # base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Flatten())
    # base_model.add(tf.keras.layers.Dense(120, activation='relu'))
    base_model.add(tf.keras.layers.Dense(50, activation='relu'))
    base_model.add(tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax))


    # hyperparams = HyperParameters(lr=1e-1, alpha =0, pi =0.3, batch_size = 128)
    # # instantiate your optimizer
    # optimizer = BBB()
    # prior = GaussianPrior(.0,-1.0)
    # prior2 = GaussianPrior(.0,-5.0)

    # optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior, prior2 = prior2)
    # optimizer.train(1000)

    hyperparams = HyperParameters(lr=3*1e-1, alpha =0, pi =0.38, batch_size = 1000)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-1.0)
    prior2 = GaussianPrior(.0,-5.0)

    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior, prior2 = prior2)
    optimizer.train(1000)

    # hyperparams = HyperParameters(lr=1e-3, k=500, frequency=1, scale=1)
    # # instantiate your optimizer
    # optimizer = SWAG()
    # optimizer.compile(hyperparams, base_model.to_json(), dataset, starting_model = base_model)
    # config = {
    #             "dataset": "CIFAR-10"
    #         }
    # optimizer.train(1000)
    # optimizer.train_with_weights_and_biases(1000, project_name="cifar10", weights_and_biases_config= config)
    

    bayesian_model: BayesianModel = optimizer.result()
    # bayesian_model.store("./MNIST_model")
    analytics_builder = Metrics(bayesian_model, dataset)
    plot_builder = Plotter(bayesian_model, dataset)
    print("Starting performance analysis")
    analytics_builder.summary(10)
    # plot_builder.confusion_matrix()
    # plot_builder.entropy()



# runner()