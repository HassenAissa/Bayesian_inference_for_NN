import numpy as np
from PyAce.nn import BayesianModel
from PyAce.datasets import Dataset
import sklearn.metrics as skmet
import scikitplot as skplt
import tensorflow as tf
import matplotlib.pyplot as plt


class Visualisation():
    """
        a class representing the performance analysis of a model
    """
    def __init__(self, model):
        self.model = model
    
    # https://seaborn.pydata.org
    def visualise(self, dataset: Dataset, nb_samples: int, loss_save_file = None):
        """
        outputs visualisations of performance metrics, learning diagnostic and uncertainty calculated upon the testing sub-dataset of given dataset. 

        Args:
            dataset (Dataset): dataset to perform analysis upon. Will use the testing sub-dataset.
            nb_samples (int): number of samples
            loss_save_file (_type_, optional): Path to file storing loss values throughout training. Defaults to None.
        """
        x, y_true = next(iter(dataset.valid_data.batch(dataset.valid_data.cardinality())))
        y_samples, y_pred = self.model.predict(x, nb_samples)  # pass in the x value
        self.learning_diagnostics(loss_save_file)

        if dataset.likelihood_model == "Regression" :
            self.pred_true_graph(x, y_true, y_pred, regression=True)
            self.uncertainty(y_samples, y_pred, y_true, regression=True)
            print("MSE:", self.mse(y_pred, y_true))
            print("RMSE:", self.rmse(y_pred, y_true))
            print("MAE:", self.mae(y_pred, y_true))
            print("R2:", self.r2(y_pred, y_true))
            
        elif dataset.likelihood_model == "Classification":
            self.pred_true_graph(x, y_true, y_pred)
            self.confusion_matrix(y_true, y_pred)
            self.uncertainty(y_samples, y_pred, y_true)
            print("Accuracy: {}%".format(self.accuracy(y_pred, y_true)))
            print("Recall: {}%".format(self.recall(y_pred, y_true)))
            print("Precision: {}%".format(self.precision(y_pred, y_true)))
            print("F1 Score: {}%".format(self.f1_score(y_pred, y_true)))
            #skplt.metrics.plot_precision_recall(y_true, y_pred, title = 'Precision-Recall Curve')
            #plt.show()
            # self.uncertainty_classification(y_samples)
        else: 
            print("Invalid loss function")
            
    def pred_true_graph(self, x, y_true, y_pred, regression=False):
        if regression:
            y_true = tf.reshape(y_true, y_pred.shape)
            if y_true.shape[1] == 1:
                plt.figure(figsize=(10, 5))
                plt.scatter(range(len(y_true)), y_true, label='True Values', alpha=0.5)
                plt.scatter(range(len(y_pred)), y_pred, label='Predicted Mean', alpha=0.5)
                plt.legend()
                plt.title('True vs Predicted Values')
                plt.xlabel('Sample Index')
                plt.ylabel('Output')
                plt.show()
        else:
            y_pred_labels = tf.argmax(y_pred, axis=1)
            x_2d = tf.reshape(x, (x.shape[0], -1))
            _, eigenvectors = tf.linalg.eigh(tf.tensordot(tf.transpose(x_2d), x_2d, axes=1))
            x_pca = tf.tensordot(x_2d, eigenvectors, axes=1)
            self.plot_2d(x_pca, y_true, y_pred_labels)
            if(x_pca.shape[1]>=3):
                self.plot_3d(x_pca, y_true, y_pred_labels)
            
        
    # Regression performance metrics
    
    def mse(self, y_pred, y_true):
        return skmet.mean_squared_error(y_true, y_pred)
    
    def rmse(self, y_pred, y_true):
        return skmet.mean_squared_error(y_true, y_pred, squared=False)
    
    def mae(self, y_pred, y_true):
        return skmet.mean_absolute_error(y_true, y_pred)
    
    def r2(self, y_pred, y_true):
        return skmet.r2_score(y_true, y_pred)
            
        
    # Classification performance metrics
    
    def accuracy(self, y_pred, y_true):
        return skmet.accuracy_score(y_true, tf.argmax(y_pred, axis = 1)) * 100
    
    def precision(self, y_pred, y_true):
        return skmet.recall_score(y_true, tf.argmax(y_pred, axis = 1), average= "macro") * 100
    
    def recall(self, y_pred, y_true):
        return skmet.precision_score(y_true, tf.argmax(y_pred, axis = 1), average= "micro") * 100
    
    def f1_score(self, y_pred, y_true):
        return skmet.f1_score(y_true, tf.argmax(y_pred,axis = 1), average = "macro") * 100
    
    def confusion_matrix(self, y_true, y_pred):
        y_pred_labels = tf.argmax(y_pred, axis=1)
        skplt.metrics.plot_confusion_matrix(y_true, y_pred_labels, normalize=True, title = 'Confusion Matrix')
        plt.show()
    
    def plot_2d(self, x_pca, y_true, y_pred):
        fig, (ax_true, ax_pred) = plt.subplots(2, figsize=(12, 8))
        scatter_true = ax_true.scatter(x_pca[:, -1], x_pca[:, -2], c=y_true, s=5)
        legend_plt_true = ax_true.legend(*scatter_true.legend_elements(), loc="lower left", title="Digits")
        ax_true.add_artist(legend_plt_true)
        scatter_pred = ax_pred.scatter(x_pca[:, -1], x_pca[:, -2], c=y_pred, s=5)
        legend_plt_pred = ax_pred.legend(*scatter_pred.legend_elements(), loc="lower left", title="Digits")
        ax_pred.add_artist(legend_plt_pred)
        ax_true.set_title('First Two Dimensions of Projected True Data After Applying PCA')
        ax_pred.set_title('First Two Dimensions of Projected Predicted Data After Applying PCA')
        plt.show()
        
    def plot_3d(self, x_pca, y_true, y_pred):
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax_true = fig.add_subplot(1, 2, 1, projection='3d')
        plt_3d_true = ax_true.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=y_true, s=1)
        fig.colorbar(plt_3d_true, shrink=0.5)
        
        ax_pred = fig.add_subplot(1, 2, 2, projection='3d')
        plt_3d_pred = ax_pred.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=y_pred, s=1)
        fig.colorbar(plt_3d_pred, shrink=0.5)
        
        plt.title('First Three Dimensions of Projected True Data (left) VS Predicted Data (right) After Applying PCA')
        plt.show()
        
    def uncertainty(self, y_samples, y_pred, y_true, regression=False) -> tuple:
        # For classification, we might use the entropy of the predicted probabilities
        # as a measure of aleatoric uncertainty and variance of multiple stochastic
        # forward passes as epistemic uncertainty.

        # Assuming predict returns a distribution over classes for each sample
        if regression:
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
            plt.show()
        else:
            mean = np.mean(y_samples, axis=0)
            variance = np.var(y_samples, axis=0)
            print("""Uncertainty for Classificaton: 
                Epistemic Uncertainty: {}
                Aleatoric Uncertainty: {}""".format(variance, mean))
        
    def learning_diagnostics(self, loss_file):
        if loss_file != None:
            losses = np.loadtxt(loss_file)
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()