import tensorflow as tf
from PIL import Image
from src.datasets.Dataset import Dataset
from src.nn.BayesianModel import BayesianModel
from src.optimizers.HyperParameters import HyperParameters
from src.optimizers.SWAG import SWAG
import numpy as np
import os
from src.visualisations.Visualisation import Visualisation



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


x, _, y,_ = load_data(r"...")
x = x.reshape(x.shape[0], -1)

# normalize, add bias
means = np.mean(x, axis=0, keepdims=True)
stds = np.std(x, axis=0, keepdims=True)

x = normalize_fn(x, means, stds)
fraction_train = 0.1
n_samples = x.shape[0]
rinds = np.random.permutation(n_samples)

n_train = int(n_samples * fraction_train)


x = x[rinds[:n_train]]
y = y[rinds[:n_train]]
n_classes = get_n_classes(y)
x = x.reshape(x.shape[0], 32, 32, 1)
dataset = tf.data.Dataset.from_tensor_slices((x, y))

dataset = Dataset(
    dataset,
    tf.keras.losses.SparseCategoricalCrossentropy(),
    "Classification"
)




base_model = tf.keras.Sequential()

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
optimizer.compile(hyperparams, base_model.get_config(), dataset, starting_model=base_model)
optimizer.train(300)


bayesian_model: BayesianModel = optimizer.result()

_, prediction = bayesian_model.predict(x, 100)
compare = tf.equal(tf.math.argmax(prediction, axis=-1), y)
print("Accuracy of the bayesian model :", tf.reduce_sum(tf.cast(compare, tf.float32) / x.shape[0]).numpy())
path = r"..."

bayesian_model.store(path)
bayesian_model: BayesianModel= BayesianModel.load(path)

_, prediction = bayesian_model.predict(x, 100)

compare = tf.equal(tf.math.argmax(prediction, axis=-1), y)

print("Accuracy of the bayesian model after store/load :", tf.reduce_sum(tf.cast(compare, tf.float32) / x.shape[0]).numpy())

analytics_builder = Visualisation(bayesian_model)

analytics_builder.visualise(dataset, 100)