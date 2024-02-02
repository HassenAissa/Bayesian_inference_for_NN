import tensorflow as tf
from tensorflow.keras import models, layers

from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG
from src.visualisations.Visualisation import Visualisation
import tensorflow_datasets as tfds


dataset = Dataset(
    "mnist",
    tf.keras.losses.SparseCategoricalCrossentropy(),
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

hyperparams = HyperParameters(lr=1e-3, k=50, frequency=1, scale=1)
# instantiate your optimizer
optimizer = SWAG()

# compile the optimizer with your data
# this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
optimizer.compile(hyperparams, base_model.get_config(), dataset, starting_model=base_model)

loss_save_file = r"package/src/visualisations/loss_save_file"
optimizer.train(1000, loss_save_file)

bayesian_model: BayesianModel = optimizer.result()
# store_path = r"..."
# bayesian_model.store(store_path)
# bayesian_model: BayesianModel= BayesianModel.load(store_path)

analytics_builder = Visualisation(bayesian_model)

analytics_builder.visualise(dataset, 100, loss_save_file)


