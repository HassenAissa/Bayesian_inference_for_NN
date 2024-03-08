from Pyesian.optimizers import SGLD
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from Pyesian.datasets import Dataset
from Pyesian.nn import BayesianModel
from Pyesian.optimizers import HMC
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.optimizers import SWAG
from Pyesian.visualisations import Metrics, Plotter
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def runner():

    x,y = sklearn.datasets.make_moons(n_samples=2000)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.BinaryCrossentropy,
        "Classification"
    )

    base_model = tf.keras.Sequential()

    base_model.add(layers.Dense(50, activation='tanh', input_shape=(2,)))
    base_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

    # hyperparams = HyperParameters(lr=1e-3, k=50, frequency=1, scale=1)
    # # instantiate your optimizer
    # optimizer = SWAG()¨
    # # compile the optimizer with your data
    # # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    # optimizer.compile(hyperparams, base_model.get_config(), dataset, starting_model=base_model)


    hyperparams = HyperParameters(lr=3, alpha = 0.00, pi = 1, batch_size = 1000)
    # instantiate your optimizer
    optimizer = BBB()
    prior = GaussianPrior(.0,-3.0)


    # prior = GuassianPrior(0.0,2.01)
    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior = prior)



    optimizer.train(2)
    bayesian_model: BayesianModel = optimizer.result()

    # bayesian_model.store("testmodel")

    # bayesian_model = BayesianModel.load("testmodel1sample")
    plotter = Plotter(bayesian_model, dataset)
    plotter.roc_one_vs_rest(label_of_interest=1)
    # plotter.plot_decision_boundaries(granularity=0.0001, un_zoom_level=1)
    # plotter.plot_uncertainty_area(uncertainty_threshold=0.9, un_zoom_level=1)
    # plotter.compare_prediction_to_target()
    # plotter.confusion_matrix()
    metrics = Metrics(bayesian_model, dataset)
    metrics.summary(100)
    # play and compate with one NN.

runner()
