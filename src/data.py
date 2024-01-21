import numpy as np
from PIL import Image
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
