from Pyesian.optimizers.hyperparameters import WandbHyperparametersOptimizer
from Pyesian.distributions import GaussianPrior
from Pyesian.optimizers import BBB
import tensorflow as tf
from tensorflow.keras import models, layers

from Pyesian.datasets import Dataset
from Pyesian.nn import BayesianModel
from Pyesian.optimizers.hyperparameters import HyperParameters
from Pyesian.optimizers import SWAG
from Pyesian.visualisations import Metrics
import tensorflow_datasets as tfds
import sklearn
import tensorflow_probability as tfp

def runner():
    x,y = sklearn.datasets.make_moons(n_samples=2000)
    dataset = Dataset(
        tf.data.Dataset.from_tensor_slices((x, y)),
        tf.keras.losses.SparseCategoricalCrossentropy(),
        "Classification"    
    )
    # dataset = Dataset(r"C:\Users\hasse\Documents\Hassen\SEGP\Datasets\Boston.csv",tf.keras.losses.MeanSquaredError(), "Regression", feature_normalisation=True)

    base_model = tf.keras.Sequential()
    base_model.add(layers.Dense(10, activation='relu', input_shape=(2,)))
    base_model.add(tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax))

    # instantiate your optimizer
    sweep_config = {
          'method': 'random', 
            'metric': {
                'name': 'loss',
                'goal': 'minimize'
            },
            'early_terminate':{
                'type': 'hyperband',
                'min_iter': 100
            },
            'parameters': {
                'epochs': {
                    'values': [100,200,300]
                },
                'lr':{
                    'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
                },
                'k':{
                    'values': [1]
                },
                'scale':{
                    'values': [1]
                },
                'frequency':{
                    'values': [1]
                }
            }
    }

    # prior = GuassianPrior(0.0,2.01)
    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    hyperparameters_selector = WandbHyperparametersOptimizer(dataset, SWAG, base_model.to_json, starting_model = base_model)
    hyperparameters_selector.hyper_parameter_tuning_with_weights_and_biases(sweep_config, count = 4, project_name="hyper")


# runner()
