import tensorflow as tf

from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, model: BayesianModel, dataset: Dataset):
        self._dataset = dataset
        self._model: BayesianModel = model

    def plot_decision_boundaries(self, dimension=2, granularity=1e-2, n_boundaries=10, n_samples=100,
                                 data_type="test", un_zoom_level=0.2):
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("Decision boundary can only be plotted for Classification")
        x, y, base_matrix = self._extract_x_y_from_dataset(dimension=dimension, n_samples=n_samples,
                                                           data_type=data_type)
        if dimension == 2:
            self._plot_2d_decision_boundary(x, y, base_matrix, dimension=2, granularity=1e-2, n_boundaries=10, un_zoom_level=un_zoom_level)

    def plot_uncertainty_area(self,
                              dimension=2,
                              granularity: float = 1e-2,
                              n_samples=100, data_type="test", uncertainty_threshold=0.8,
                              un_zoom_level=0.2):
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("Uncertainty area can only be plotted for Classification")
        x, y, base_matrix = self._extract_x_y_from_dataset(dimension=dimension, n_samples=n_samples,
                                                           data_type=data_type)
        if dimension == 2:
            self._plot_2d_uncertainty_area(x, y, base_matrix, granularity, n_samples, uncertainty_threshold,
                                           un_zoom_level)

    def _plot_2d_uncertainty_area(self,
                                  x: tf.Tensor,
                                  y: tf.Tensor,
                                  base_matrix: tf.Tensor,
                                  granularity: float,
                                  n_samples: int,
                                  uncertainty_threshold: float,
                                  un_zoom_level: float):
        dim1, dim2, grid_x_augmented = self._extract_grid_x(x, base_matrix, granularity, un_zoom_level)
        _, predictions = self._model.predict(grid_x_augmented, n_samples)
        n_classes = tf.unique(y)[0].shape[0]
        colors = [(i / n_classes + 0.5 / n_classes) for i in range(n_classes)]
        for i in range(n_classes):
            plt.scatter(x[y == i][:, 0], x[y == i][:, 1], marker='o', cmap=colors[i], label="Class " + str(i))
        if predictions.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            predictions = tf.stack([predictions, 1 - predictions], axis=1)
        predictions_max = tf.math.reduce_max(predictions, axis=1)
        uncertainty_area = tf.cast(predictions_max < uncertainty_threshold, dtype=tf.float32)
        uncertainty_area = tf.reshape(uncertainty_area, (dim1.shape[0], dim1.shape[1]))
        plt.contourf(dim1, dim2, uncertainty_area, [0.9, 1.1], colors=["orange"], alpha=0.5)
        plt.legend()
        plt.title("Uncertainty area with threshold " + str(uncertainty_threshold))
        plt.show()

    def _extract_x_y_from_dataset(self, dimension=2, n_samples=100, data_type="test") -> (tf.Tensor, tf.Tensor):
        tf_dataset = self._dataset.valid_data
        if data_type == "test":
            tf_dataset = self._dataset.test_data
        elif data_type == "train":
            tf_dataset = self._dataset.train_data
        if n_samples > tf_dataset.cardinality().numpy().item():
            n_samples = tf_dataset.cardinality().numpy().item()
            print("Warning : n_samples is larger than specified dataset. Will plot less samples")
        x, y = next(iter(tf_dataset.batch(n_samples)))

        if x.shape[1] > dimension:
            print("Dimension ", len(x.shape[1]), " is not right.")
            print("Will apply PCA to reduce to dimension ", dimension)
            base_matrix = tf.pca(x, dimension, dtype=x.dtype)
            return tf.linalg.matmul(x, base_matrix), y, base_matrix
        elif x.shape[1] < dimension:
            raise ValueError("Dimension ", x.shape[1], " is inferior to given dimension")
        return x, y, tf.eye(dimension, dtype=x.dtype)

    def _plot_2d_decision_boundary(self,
                                   x: tf.Tensor,
                                   y: tf.Tensor,
                                   base_matrix: tf.Tensor,
                                   dimension=2,
                                   granularity=1e-2,
                                   n_boundaries=10,
                                   un_zoom_level=0.2):
        dim1, dim2, grid_x_augmented = self._extract_grid_x(x, base_matrix, granularity, un_zoom_level)
        prediction_samples, _ = self._model.predict(grid_x_augmented, n_boundaries)
        plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='o', c="blue", label="Class 0")
        plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='x', c="red", label="Class 1")
        for pred in prediction_samples:
            pred = tf.reshape(pred, dim1.shape)
            plt.contour(dim1, dim2, pred, [0.5], colors=["red"])
        plt.legend()
        plt.title("Multiple Decision Boundaries N=" + str(n_boundaries))
        plt.show()

    def _extract_grid_x(self, x, base_matrix, granularity, un_zoom_level: float):
        max_features = tf.math.reduce_max(x, axis=0)
        min_features = tf.math.reduce_min(x, axis=0)
        size1 = (max_features[0] - min_features[0])
        size2 = (max_features[1] - min_features[1])
        dim1 = tf.range(min_features[0] - (un_zoom_level/2) * size1,
                        max_features[0] + (un_zoom_level/2) * size1,
                        granularity * (max_features[0] - min_features[0] + un_zoom_level * size1))
        dim2 = tf.range(min_features[1] - (un_zoom_level/2) * size2,
                        max_features[1] + (un_zoom_level/2) * size2,
                        granularity * (max_features[1] - min_features[1] + un_zoom_level * size2))
        dim1, dim2 = tf.meshgrid(dim1, dim2, indexing='ij')
        grid_x = tf.stack([tf.reshape(dim1, (-1)), tf.reshape(dim2, (-1))], axis=1)
        grid_x_augmented = tf.linalg.matmul(grid_x, tf.transpose(base_matrix))
        return dim1, dim2, grid_x_augmented
