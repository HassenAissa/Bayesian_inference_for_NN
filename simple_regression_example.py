import tensorflow as tf

from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import BBB, SGD
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.nn import BayesianModel
from Pyesian.visualisations import Metrics

# Create a dummy dataset
x = tf.random.uniform(shape=(600,1), minval=1, maxval=20, dtype=tf.float32)
y = 2*x+2
# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.MeanSquaredError,
    "Regression"
)

# Create your tf.keras model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, activation='linear', input_shape=(1,)))

# Indicate your hyperparameters
hyperparams = HyperParameters(lr=1e-3, frequency=1)
# Instantiate your optimizer
optimizer = SGD()
# Compile the optimizer with your data and the training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, starting_model=model)
optimizer.train(2000)
# You are done! Here is your BayesianModel
bayesian_model: BayesianModel = optimizer.result()

# See your metrics and performance
metrics = Metrics(bayesian_model, dataset)
metrics.summary()
# Save your model to a folder
bayesian_model.store("sgd-saved")