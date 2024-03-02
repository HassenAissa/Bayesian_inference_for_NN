import math

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HMC
from PyAce.optimizers.hyperparameters import HyperParameters
import tensorflow as tf
from PyAce.optimizers.hyperparameters import GridOptimizer

from PyAce.distributions import GaussianPrior
from PyAce.optimizers.hyperparameters.space import Real, Integer

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
dataset = Dataset("mnist", tf.keras.losses.SparseCategoricalCrossentropy(), "Classification", feature_normalisation=True, train_proportion=0.03, valid_proportion=0.03, test_proportion=0.94)



def f(epsilon, L, m):
    hyperparams = HyperParameters(epsilon=epsilon, L=L, m=m, nb_burn_epoch=1)
    # instantiate your optimizer
    prior = GaussianPrior(
        0.0, 1.0
    )
    optimizer = HMC()
    optimizer.compile(hyperparams, base_model.to_json(), dataset, prior=prior, verbose=False)
    optimizer.train(1)
    bayesian_model: BayesianModel = optimizer.result()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    x, y = next(iter(dataset.valid_data.batch(dataset.valid_data.cardinality())))
    _, y_pred = bayesian_model.predict(x, 1)
    print(x)
    print(y_pred)
    #print(y_pred)
    return math.sqrt(loss(y,y_pred))



optimizer = GridOptimizer()


optimizer.compile(f,
                  Real(1e-3, 1e-1, "epsilon"),
                  Integer(10, 300, "L"),
                  Real(1e-1, 3, "m"),
                  specify={"epsilon": [1e-4,0.0005 ,1e-3,0.005,1e-2,0.05, 1e-1],
                           "L": [10,30,50,100,300],
                           "m": [1,2]
                           }, n=10)


'''
optimizer.compile(f,
                  Real(1e-3, 1e-1, "epsilon"),
                  Integer(10, 100, "L"),
                  Real(1e-1, 3, "m"),
                  specify={"epsilon": [1e-5],
                           "L": [500],
                           "m": [3]
                           }, n=10)
'''
optimizer.optimize(nb_processes=2)
optimizer.save("mnist.txt")

