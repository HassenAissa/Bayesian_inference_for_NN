import tensorflow as tf

from Pyesian.datasets import Dataset
from Pyesian.nn import BayesianModel
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from sklearn.decomposition import PCA
import os
import sklearn as sk



class Plotter:
    """
        a class giving visualisation tools for performance analysis of a model over a dataset
        Args:
            model (BayesianModel): trained model that will make the predictions
            dataset (Dataset): the dataset on which to calculate the metrics
    """
    def __init__(self, model: BayesianModel, dataset: Dataset):
        self._dataset = dataset
        self._model: BayesianModel = model
        self._nb_predictions: int = 0
        self._cached_samples : list = None
        self._cached_prediction: tf.Tensor = None
        self._cached_true_values: tf.Tensor = None
        self._cached_input: tf.tensor = None
        self._cached_data_type = None

    
    def _get_predictions(self, input, nb_boundaries, y_true, data_type):
        if (self._nb_predictions == nb_boundaries 
            and y_true.shape == self._cached_true_values.shape
            and data_type == self._cached_data_type):
            y_pred = self._cached_prediction
            if self._cached_prediction.shape[1] == 1 and self._dataset.likelihood_model == "Classification":
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1 - self._cached_prediction, self._cached_prediction], axis=1)
            return self._cached_samples, y_pred, self._cached_true_values, self._cached_input
        else:
            y_samples, y_pred = self._model.predict(input, nb_boundaries)  # pass in the x value
            self._nb_predictions = nb_boundaries
            self._cached_data_type = data_type
            self._cached_input = input
            self._cached_samples = tf.identity(y_samples)
            self._cached_prediction = tf.identity(y_pred)
            self._cached_true_values = tf.identity(y_true)
            if y_pred.shape[1] == 1 and self._dataset.likelihood_model == "Classification":
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
            return y_samples, y_pred, y_true, input

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
            predictions = tf.stack([1 - predictions, predictions], axis=1)
        predictions_max = tf.math.reduce_max(predictions, axis=1)
        uncertainty_area = tf.cast(predictions_max < uncertainty_threshold, dtype=tf.float32)
        uncertainty_area = tf.reshape(uncertainty_area, (dim1.shape[0], dim1.shape[1]))
        plt.contourf(dim1, dim2, uncertainty_area, [0.9, 1.1], colors=["orange"], alpha=0.5)
        plt.legend()
        plt.title("Uncertainty area with threshold " + str(uncertainty_threshold))

    def _get_x_y(self, n_samples=100, data_type="test"):
        tf_dataset = self._dataset.valid_data
        if data_type == "test":
            tf_dataset = self._dataset.test_data
        elif data_type == "train":
            tf_dataset = self._dataset.train_data
        x,y_true = next(iter(tf_dataset.batch(n_samples)))
        return x,y_true

    def _extract_x_y_from_dataset(self, dimension=2, n_samples=100, data_type="test") -> (tf.Tensor, tf.Tensor):
        x, y = self._get_x_y(n_samples, data_type)

        if x.shape[1] > dimension: #TODO: does not take images into account
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
    
    def roc_one_vs_rest(self, n_samples = 100, label_of_interest: int = 0, n_boundaries = 10, data_type = "test"):
        """plots the ROC curve in a one vs the rest fashion

        Args:
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            label_of_interest (int, optional): The label that will be opposed to the other for ROC calculation. Defaults to 0.
            n_boundaries (int, optional): the number of sampled networks for the Monte Carlo approximation. Defaults to 10.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".

        Raises:
            ValueError: when the method is called for other than a classification problem
        """
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("ROC can only be plotted for Classification")  
        x,y_true = self._get_x_y(n_samples, data_type)
        y_samples, y_pred, y_true, x = self._get_predictions(x, n_boundaries, y_true, data_type)
        one_hot_y_true = tf.one_hot(y_true, y_pred.shape[1])
        display = sk.metrics.RocCurveDisplay.from_predictions(
            one_hot_y_true[:, label_of_interest],
            y_pred[:, label_of_interest],
            name=f"class {label_of_interest} vs the rest",
            color="blue",
            plot_chance_level=True,
        )
        _ = display.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="ROC curve One-vs-Rest",
        )
        plt.show()

   
    def plot_decision_boundaries(self, dimension=2, granularity=1e-2, n_boundaries=30, n_samples=100,
                                 data_type="test", un_zoom_level=0.2, save_path=None):
        """plots the n_boundaries decision boundaries for a classification task

        Args:
            dimension (int, optional): the dimension of the feature space for the plot. Defaults to 2.
            granularity (int, optional): The precision of the plot. Defaults to 1e-2.
            n_boundaries (int, optional): the number of sampled networks for the Monte Carlo approximation each one gives a new decision boundary. Defaults to 30.
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            un_zoom_level (float, optional): The zoom level for the plot. Defaults to 0.2.
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.

        Raises:
            ValueError: when the method is called for other than a classification problem

        """
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("Decision boundary can only be plotted for Classification")
        x, y, base_matrix = self._extract_x_y_from_dataset(dimension=dimension, n_samples=n_samples,
                                                           data_type=data_type)
        if dimension == 2:
            self._plot_2d_decision_boundary(x, y, base_matrix, dimension=2, granularity= granularity, 
                                            n_boundaries=10, un_zoom_level=un_zoom_level)
            self._save(save_path, "decision_boundaries") if save_path else plt.show()


    def plot_uncertainty_area(self,
                              dimension=2,
                              granularity: float = 1e-2,
                              n_samples=100, data_type="test", uncertainty_threshold=0.8,
                              un_zoom_level=0.2,
                              save_path=None):
        """plots the uncertainty area for a classification task

        Args:
            dimension (int, optional): the dimension of the feature space for the plot. Defaults to 2.
            granularity (float, optional): The precision of the plot. Defaults to 1e-2.
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            uncertainty_threshold (float, optional): the threshold below which we consider the prediction uncertain. Defaults to 0.8.
            un_zoom_level (float, optional): The zoom level for the plot. Defaults to 0.2.
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.

        Raises:
            ValueError: when the method is called for other than a classification problem
        """
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("Uncertainty area can only be plotted for Classification")
        x, y, base_matrix = self._extract_x_y_from_dataset(dimension=dimension, n_samples=n_samples,
                                                           data_type=data_type)
        if dimension == 2:
            self._plot_2d_uncertainty_area(x, y, base_matrix, granularity, n_samples, uncertainty_threshold,
                                           un_zoom_level)
            self._save(save_path, "uncertainty_area") if save_path else plt.show()


    def regression_uncertainty(self, n_boundaries = 30, n_samples = 100, data_type="test", save_path=None) -> tuple:
        """Plots the epistemic uncertainty for a regression problem

        Args:
            n_boundaries (int, optional): the number of sampled networks for the Monte Carlo approximation each one gives a new decision boundary. Defaults to 30.
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.

        Raises:
            ValueError: when the method is called for other than a regression dataset

        """
        if self._dataset.likelihood_model == "Regression":
            x,y_true = self._get_x_y(n_samples, data_type)
            y_samples, y_pred, y_true,x = self._get_predictions(x, n_boundaries, y_true, data_type)
            variance = np.var(y_samples, axis=0)
            err = np.mean(np.sqrt(variance), axis = 1)
            pred_dev = np.mean((y_pred.numpy()-y_true.numpy()), axis = 1)
            # uncertainty
            plt.figure(figsize=(10, 5))
            plt.hlines([0], 0, len(err))
            plt.plot(range(len(err)), pred_dev-err, label='Epistemic Lower', alpha=0.5)
            plt.scatter(range(len(err)), pred_dev, label='Averaged deviation', alpha=0.5, c="k")
            plt.plot(range(len(err)), pred_dev+err, label='Epistemic Upper', alpha=0.5)
            plt.legend()
            plt.title('Epistemic Uncertainty')
            plt.ylabel('Pred-True difference')
            self._save(save_path, "epistemic_uncertainty") if save_path else plt.show()
        else:
            raise ValueError("regression uncertainty cannot be computed for other than regression problems")

    

    def confusion_matrix(self, n_boundaries = 30, n_samples=100, data_type="test",  save_path=None):
        """Plots the confusion matrix for a classification problem

        Args:
            n_boundaries (int, optional): the number of sampled networks for the Monte Carlo approximation each one gives a new decision boundary. Defaults to 30.
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.

        Raises:
            ValueError: when the method is called for other than a classification problem

        """
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("Confusion matrix cannot be computed for other than classification problems")

        x,y_true = self._get_x_y(n_samples, data_type)
        y_samples, y_pred, y_true,x = self._get_predictions(x, n_boundaries, y_true, data_type)
        y_pred_labels = tf.argmax(y_pred, axis=1)
        y_true = tf.reshape(y_true, y_pred_labels.shape)
        skplt.metrics.plot_confusion_matrix(y_true, y_pred_labels, normalize=True, title = 'Confusion Matrix')
        self._save(save_path, "confusion_matrix") if save_path else plt.show()
            

    def compare_prediction_to_target(self, n_boundaries = 30, n_samples=100, data_type="test", save_path=None):
        """
        Plots a comparision between the true values or labels and the predicitions for both classification and regression problems.

        Args:
            n_boundaries (int, optional): the number of sampled networks for the Monte Carlo approximation each one gives a new decision boundary. Defaults to 30.
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.
        """
        x,y_true = self._get_x_y(n_samples, data_type)
        y_samples, y_pred, y_true, x = self._get_predictions(x, n_boundaries, y_true, data_type)
        if self._dataset.likelihood_model == "Regression":
            y_true = tf.reshape(y_true, y_pred.shape)
            if y_true.shape[1] == 1:
                plt.figure(figsize=(10, 5))
                plt.scatter(range(len(y_true)), y_true, label='True Values', alpha=0.5)
                plt.scatter(range(len(y_pred)), y_pred, label='Predicted Mean', alpha=0.5)
                plt.legend()
                plt.title('True vs Predicted Values')
                plt.xlabel('Sample Index')
                plt.ylabel('Output')
                self._save(save_path, "comparison_pred_true") if save_path else plt.show()
        else:
            y_pred_labels = tf.argmax(y_pred, axis=1)
            x_2d = tf.reshape(x, (x.shape[0], -1))
            if x_2d.shape[1] == 2:
                self._compare_prediction_to_target_2d(x_2d, y_true, y_pred_labels)
            else:
                if(x_2d.shape[1]>=3):
                    x_pca = PCA(n_components=3).fit_transform(x_2d)
                    self._compare_prediction_to_target_3d(x_pca, y_true, y_pred_labels, save_path=save_path)
                else:
                    x_pca = PCA(n_components=2).fit_transform(x_2d)
                    self._compare_prediction_to_target_2d(x_pca, y_true, y_pred_labels, save_path=save_path)


    def _compare_prediction_to_target_2d(self, x_pca, y_true, y_pred, save_path=None):
        fig, (ax_true, ax_pred) = plt.subplots(2, figsize=(12, 8))
        scatter_true = ax_true.scatter(x_pca[:, -2], x_pca[:, -1], c=y_true, s=5)
        legend_plt_true = ax_true.legend(*scatter_true.legend_elements(), loc="lower left", title="Digits")
        ax_true.add_artist(legend_plt_true)
        scatter_pred = ax_pred.scatter(x_pca[:, -2], x_pca[:, -1], c=y_pred, s=5)
        legend_plt_pred = ax_pred.legend(*scatter_pred.legend_elements(), loc="lower left", title="Digits")
        ax_pred.add_artist(legend_plt_pred)
        ax_true.set_title('First Two Dimensions of Projected True Data After Applying PCA')
        ax_pred.set_title('First Two Dimensions of Projected Predicted Data After Applying PCA')
        self._save(save_path, "comparison_pred_true") if save_path else plt.show()
        
    def _compare_prediction_to_target_3d(self, x_pca, y_true, y_pred, save_path=None):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax_true = fig.add_subplot(1, 2, 1, projection='3d')
        plt_3d_true = ax_true.scatter3D(x_pca[:, -3], x_pca[:, -2], x_pca[:, -1], c=y_true, s=1)
        fig.colorbar(plt_3d_true, shrink=0.5)
        
        ax_pred = fig.add_subplot(1, 2, 2, projection='3d')
        plt_3d_pred = ax_pred.scatter3D(x_pca[:, -3], x_pca[:, -2], x_pca[:, -1], c=y_pred, s=1)
        fig.colorbar(plt_3d_pred, shrink=0.5)
        
        plt.title('First Three Dimensions of Projected True Data (left) VS Predicted Data (right) After Applying PCA')
        self._save(save_path, "comparison_pred_true") if save_path else plt.show()

    def entropy(self, n_boundaries = 30, n_samples=100, data_type="test", save_path=None):
        """Plots the entropy over the prediciton probabilities for a classification problem

        Args:
            n_boundaries (int, optional): the number of sampled networks for the Monte Carlo approximation each one gives a new decision boundary. Defaults to 30.
            n_samples (int, optional): number of samples from the dataset. Defaults to 100.
            data_type (str, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.

        Raises:
            ValueError: when the method is called for other than a classification problem

        """
        x,y_true = self._get_x_y(n_samples, data_type)
        y_samples, y_pred, y_true, x = self._get_predictions(x, n_boundaries, y_true, data_type)
        if self._dataset.likelihood_model == "Classification":
            entropies = []
            for probabilities in y_pred:
                entropies.append(-1*np.sum(probabilities*np.log(probabilities+1e-5)))
            entropies = np.sort(np.nan_to_num(entropies))
            plt.plot(range(len(y_true)), entropies)
            plt.title('Entropies for each input')
            plt.xlabel('Sample Index')
            plt.ylabel('entropy')
            self._save(save_path, "entropy") if save_path else plt.show()
        else:
            raise Exception("Entropy is only available for classification")
        


    def learning_diagnostics(self, loss_file: str, save_path=None):
        """plots the evolution of the loss function during the training. The losses should be saved inside a file.

        Args:
            loss_file (str): The path to the file with the training losses
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the plots. Defaults to None.
        """
        if loss_file != None:
            losses = np.loadtxt(loss_file)
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            self._save(save_path, "learning_diagnostics") if save_path else plt.show()


    def _save(self, save_path, name):
        directory = os.path.join(save_path,"report")
        plots = os.path.join(directory, "plots")
        os.makedirs(directory, exist_ok=True)
        os.makedirs(plots, exist_ok=True)
        plt.savefig(os.path.join(plots, name + ".png"))
