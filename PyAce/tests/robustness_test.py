import tensorflow as tf
from tensorflow.keras import models, layers

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import BBB
from PyAce.distributions import GaussianPrior
from PyAce.visualisations import Robustness
import tensorflow_datasets as tfds

def runner():
    dataset = Dataset(
        "cifar10",
        tf.keras.losses.SparseCategoricalCrossentropy(),
        "Classification",
    )
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    base_model = tf.keras.Sequential()

    base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32, 32, 3)))
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
    print("Starting robustness analysis")
    path = "PyAce/tests"
    robustness_builder = Robustness.Robustness(bayesian_model, dataset)
    corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
                            "gaussian_blur", "contrast", "brightness", "saturate","pixelate"]
    for c in corruptions:
        robustness_builder.corruption_error(c)
    
    robustness_builder.mean_corruption_error()
    robustness_builder.mean_corruption_error(relative=True, save_path=path)
        
runner()