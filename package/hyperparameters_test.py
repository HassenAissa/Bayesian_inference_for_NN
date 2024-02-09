from src.optimizers.HyperparametersSelector import HyperParametersSelector
from src.distributions.GaussianPrior import GaussianPrior
from src.optimizers.BBB import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG
from src.visualisations.Visualisation import Visualisation
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp


x,y = sklearn.datasets.make_moons(n_samples=2000)
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.SparseCategoricalCrossentropy(),
    "Classification"    
)

base_model = tf.keras.Sequential()
base_model.add(layers.Dense(50, activation='tanh', input_shape=(2,)))
base_model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

# instantiate your optimizer
optimizer = BBB
prior = GaussianPrior(.0,-10.0)


# prior = GuassianPrior(0.0,2.01)
# compile the optimizer with your data
# this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
hyperparameters_selector = HyperParametersSelector(dataset, lr = (1e-6,1e-4), alpha = (0.000001,0.0001))
hyperparameters_selector.cross_validation(base_model.to_json(), optimizer, prior = prior)



