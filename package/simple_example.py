from src.optimizers.BBB import BBB
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import models, layers

from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG
from src.visualisations.Visualisation import Visualisation

x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.int32)
y = 2*x+2
dataset = tf.data.Dataset.from_tensor_slices((x, y))


#train_dataset = Dataset(train_dataset, tf.keras.losses.MeanSquaredError())
dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression")
train_dataset = Dataset(dataset.train_data, tf.keras.losses.MeanSquaredError(), "Regression")

initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model = tf.keras.models.Sequential()
model.add(layers.Dense(100, activation='tanh', input_shape=(1,))) #chnage to tanh
model.add(layers.Dense(1, activation='linear'))

hyperparams = HyperParameters(lr=1e-3, alpha = 0.0)
# instantiate your optimizer
optimizer = BBB()
prior = tfp.distributions.Normal(0,10)
# compile the optimizer with your data
# this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
optimizer.compile(hyperparams, model.get_config(), train_dataset, starting_model=model, prior = prior)

optimizer.train(300)



bayesian_model: BayesianModel = optimizer.result()
# store_path = r"..."
# bayesian_model.store(store_path)
# bayesian_model: BayesianModel= BayesianModel.load(store_path)

analytics_builder = Visualisation(bayesian_model)

analytics_builder.visualise(dataset, 100)


