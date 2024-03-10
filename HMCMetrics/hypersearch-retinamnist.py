import math

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HMC, BBB, BSAM
from PyAce.optimizers.hyperparameters import HyperParameters
import tensorflow as tf
from PyAce.optimizers.hyperparameters import GridOptimizer

from PyAce.distributions import GaussianPrior
from PyAce.optimizers.hyperparameters.space import Real, Integer
from PyAce.visualisations import Metrics


base_model = tf.keras.Sequential()

#base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)))
#base_model.add(tf.keras.layers.MaxPooling2D(2))
#base_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
#base_model.add(tf.keras.layers.MaxPooling2D(2))
#base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
#base_model.add(tf.keras.layers.MaxPooling2D(2))
'''
base_model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
#base_model.add(tf.keras.layers.Dense(120, activation='relu'))
#base_model.add(tf.keras.layers.Dense(84, activation='relu'))
#base_model.add(tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax))

base_model.add(tf.keras.layers.Dense(800, activation='relu'))
base_model.add(tf.keras.layers.Dense(800, activation='relu'))
base_model.add(tf.keras.layers.Dense(800, activation='relu'))
base_model.add(tf.keras.layers.Dense(50, activation='relu'))
base_model.add(tf.keras.layers.Dense(5, activation=tf.keras.activations.softmax))
'''
from tensorflow.keras import layers
base_model = tf.keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
dataset = Dataset("Datasets/RetinaMNIST", tf.keras.losses.SparseCategoricalCrossentropy, "Classification",
                  load_images=True, train_proportion=0.7, test_proportion=0.1, valid_proportion=0.2)
import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(x, filters, downsample=False):
    y = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1 if not downsample else 2), padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(y)
    y = layers.BatchNormalization()(y)

    if downsample:
        x = layers.Conv2D(filters, kernel_size=(1, 1), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)

    out = layers.add([x, y])
    out = layers.Activation('relu')(out)
    return out


def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual blocks
    num_blocks_list = [2, 2, 2]
    filters_list = [16, 32, 64]

    for i, num_blocks in enumerate(num_blocks_list):
        for j in range(num_blocks):
            x = residual_block(x, filters_list[i], downsample=(j == 0 and i != 0))

    # Classifier part
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Model
    model = Model(inputs, outputs)
    return model

from tensorflow import keras
model = tf.keras.models.Sequential()

    # C1: (None,32,32,1) -> (None,28,28,6).
model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(28,28,1), padding='valid'))

    # P1: (None,28,28,6) -> (None,14,14,6).
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # C2: (None,14,14,6) -> (None,10,10,16).
model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

    # P2: (None,10,10,16) -> (None,5,5,16).
model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Flatten: (None,5,5,16) -> (None, 400).
model.add(layers.Flatten())

    # FC1: (None, 400) -> (None,120).
model.add(layers.Dense(120, activation='tanh'))

    # FC2: (None,120) -> (None,84).
model.add(layers.Dense(84, activation='tanh'))
    # FC3: (None,84) -> (None,10).
model.add(layers.Dense(10, activation='softmax'))
base_model = model
hyperparams = HyperParameters(lr=0.1, gam=0.1, lam=0, beta_1=0.95, beta_2=0.99999, batch_size=128, rho=0.002)

optimizer = BSAM()
optimizer.compile(hyperparams, base_model.to_json(), dataset, verbose=True, starting_model=base_model)
optimizer.train(10000)
bayesian_model: BayesianModel = optimizer.result()
metrics = Metrics(bayesian_model, dataset)
metrics.summary()
