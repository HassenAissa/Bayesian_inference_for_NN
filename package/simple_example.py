import tensorflow as tf
from tensorflow.keras import models, layers

from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG

x_train = tf.random.uniform(shape=(500,1), minval=1, maxval=20, dtype=tf.int32)
y_train = 2*x_train+2
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

x_test = tf.random.uniform(shape=(100,1), minval=1, maxval=20, dtype=tf.int32)
y_test = 2*x_test+2

train_dataset = Dataset(train_dataset, tf.keras.losses.MeanSquaredError())
initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
model = tf.keras.models.Sequential()
model.add(layers.Dense(1, activation='linear', kernel_initializer=initializer, input_shape=(1,)))
#model.add(layers.Dense(1, activation='linear', kernel_initializer=initializer))

hyperparams = HyperParameters(lr=1e-3, k=50, frequency=1, scale=1)
# instantiate your optimizer
optimizer = SWAG()
# compile the optimizer with your data
# this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
optimizer.compile(hyperparams, model.get_config(), train_dataset, starting_model=model)
for i in range(0, 100):
    # train the model
    optimizer.step()
print(optimizer._base_model(x_test[0]))
bayesian_model: BayesianModel = optimizer.result()

print(x_test[0], bayesian_model.predict(x_test[0], 100))
