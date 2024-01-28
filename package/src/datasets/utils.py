import numpy as np
import os
from PIL import Image
import tensorflow as tf

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

def imgdata_preprocess(directory, fraction_train, input_shape: tuple):
    x, _, y,_ = load_data(directory)
    x = x.reshape(x.shape[0], -1)

    means = np.mean(x, axis=0, keepdims=True)
    stds = np.std(x, axis=0, keepdims=True)
    x = normalize_fn(x, means, stds)
    n_samples = x.shape[0]
    rinds = np.random.permutation(n_samples)
    n_train = int(n_samples * fraction_train)

    x = x[rinds[:n_train]]
    y = y[rinds[:n_train]]
    x = x.reshape(x.shape[0], *input_shape)
    return x,y


