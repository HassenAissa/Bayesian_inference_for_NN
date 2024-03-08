import sklearn
import tensorflow as tf
from Pyesian.datasets import Dataset
from Pyesian.distributions import GaussianPrior
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import BBB
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.visualisations import Metrics, Plotter

# Import dataset from sklearn
x,y = sklearn.datasets.make_moons(n_samples=2000)
# Wrap it in the Dataset class and indicate your loss
dataset = Dataset(
    tf.data.Dataset.from_tensor_slices((x, y)),
    tf.keras.losses.SparseCategoricalCrossentropy,
    "Classification"
)

# Create your tf.keras model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=(2,)))
model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

# Create the Prior distribution
prior = GaussianPrior(0.0, -1.0)
# Indicate your hyperparameters
hyperparams = HyperParameters(lr=0.5, alpha=0.0, batch_size=1000)
# Instantiate your optimizer
optimizer = BBB()
# Provide the optimizer with the training data and training parameters
optimizer.compile(hyperparams, model.to_json(), dataset, prior=prior)
optimizer.train(600)
# You're done ! Here is your trained BayesianModel !
bayesian_model: BayesianModel = optimizer.result()

# See your metrics and performance
metrics = Metrics(bayesian_model, dataset)
metrics.summary()
# Save your model to a folder
bayesian_model.store("bbb-saved")

# Visualize your results
plotter = Plotter(bayesian_model, dataset)
# Plot some distribution boundaries sampled from your posterior
plotter.plot_decision_boundaries(n_samples=100)
# Plot the uncertainty area of your model
plotter.plot_uncertainty_area(uncertainty_threshold=0.9)
