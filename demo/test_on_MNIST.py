import tensorflow as tf
from PIL import Image
from src.dataset.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG
import numpy as np
import os


def _load_images_from_directory(dir):
    images = []
    for imagestr in sorted(os.listdir(dir)):
        if imagestr.endswith('.png'):
            image = Image.open(os.path.join(dir, imagestr)).convert('L')
            images.append(np.asarray(image) / 255.)
    return np.array(images)


def load_data(directory):
    """
    Return the dataset as numpy arrays.

    Arguments:
        directory (str): path to the dataset directory
    Returns:
        train_images (array): images of the train set, of shape (N,H,W)
        test_images (array): images of the test set, of shape (N',H,W)
        train_labels (array): labels of the train set, of shape (N,)
        test_labels (array): labels of the test set, of shape (N',)
    """
    train_images = _load_images_from_directory(os.path.join(directory, 'train_images'))
    test_images = _load_images_from_directory(os.path.join(directory, 'test_images'))

    train_labels = np.loadtxt(os.path.join(directory, 'train_labels.csv'), dtype=int)
    test_labels = np.loadtxt(os.path.join(directory, 'test_labels.csv'), dtype=int)

    return train_images, test_images, train_labels, test_labels


def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.

    Arguments:
        data (array): of shape (N,D)
        means (array): of shape (1,D)
        stds (array): of shape (1,D)
    Returns:
        (array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds


def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.

    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


xtrain, xtest, ytrain, ytest = load_data(r"path")
xtrain = xtrain.reshape(xtrain.shape[0], -1)
xtest = xtest.reshape(xtest.shape[0], -1)

# normalize, add bias
means = np.mean(xtrain, axis=0, keepdims=True)
stds = np.std(xtrain, axis=0, keepdims=True)

xtrain = normalize_fn(xtrain, means, stds)
xtest = normalize_fn(xtest, means, stds)
fraction_train = 0.1
n_samples = xtrain.shape[0]
rinds = np.random.permutation(n_samples)

n_train = int(n_samples * fraction_train)
xtest = xtrain[rinds[n_train:]]
ytest = ytrain[rinds[n_train:]]

xtrain = xtrain[rinds[:n_train]]
ytrain = ytrain[rinds[:n_train]]
n_classes = get_n_classes(ytrain)
xtrain = xtrain.reshape(xtrain.shape[0], 32, 32, 1)
xtest = xtest.reshape(xtest.shape[0], 32, 32, 1)
train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))

train_dataset = Dataset(
    train_dataset,
    tf.keras.losses.SparseCategoricalCrossentropy()
)


base_model = tf.keras.Sequential()
#base_model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=(1024,)))
#base_model.add(tf.keras.layers.Dense(10, activation='softmax'))
#base_model.compile(optimizer=tf.keras.optimizers.SGD(lr = 1e-1), loss=tf.keras.losses.SparseCategoricalCrossentropy())
#base_model.fit(xtrain, ytrain, epochs=100)
#prediction = base_model.predict(xtest, 100)
#compare = tf.equal(tf.math.argmax(prediction, axis=-1), ytest)
#print(tf.reduce_sum(tf.cast(compare, tf.float32) / xtest.shape[0]))
#
base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32, 32, 1)))
base_model.add(tf.keras.layers.MaxPooling2D(2))
base_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
base_model.add(tf.keras.layers.MaxPooling2D(2))
base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
base_model.add(tf.keras.layers.MaxPooling2D(2))
base_model.add(tf.keras.layers.Flatten())
base_model.add(tf.keras.layers.Dense(120, activation='relu'))
base_model.add(tf.keras.layers.Dense(84, activation='relu'))
base_model.add(tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax))

hyperparams = HyperParameters(lr=1e-1, k=10, frequency=1, scale=1)
# instantiate your optimizer
optimizer = SWAG()
# compile the optimizer with your data
# this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
optimizer.compile(hyperparams, base_model.get_config(), train_dataset, starting_model=base_model)
for i in range(0, 300):
    # train the model
    optimizer.step()

bayesian_model: BayesianModel = optimizer.result()

_, prediction = bayesian_model.predict(xtest, 100)
compare = tf.equal(tf.math.argmax(prediction, axis=-1), ytest)
print(tf.reduce_sum(tf.cast(compare, tf.float32) / xtest.shape[0]))