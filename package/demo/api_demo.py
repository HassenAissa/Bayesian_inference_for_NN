import tensorflow as tf
from tensorflow.keras import models, layers

from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG

# import training testing data ect...
X_train, X_test, y_train, y_test = None, None, None, None
dataset = None #TODO: Implement this class
# Build a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

hyperparams = HyperParameters(lr=1e-2, k=50)
# instantiate your optimizer
optimizer = SWAG()
# compile the optimizer with your data
# this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
optimizer.compile(hyperparams, model.get_config(), dataset, starting_model=model)
for i in range(0, 100):
    # train the model
    optimizer.step()

# get the final probability distribution
bayesian_model : BayesianModel = optimizer.result()
# save the model to the disk
bayesian_model.store("path/to/the/folder")
# reload a saved Model
bayesian_model_loaded = BayesianModel.load("path/to/the/folder")
