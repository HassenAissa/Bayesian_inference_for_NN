import numpy as np
from src.nn.BayesianModel import BayesianModel
from src.datasets.Dataset import Dataset
import sklearn.metrics as skmet
import scikitplot as skplt
import tensorflow as tf
import matplotlib.pyplot as plt


class Visualisation():
    def __init__(self, model):
        self.model = model
    
    # https://seaborn.pydata.org
    def visualise(self, dataset: Dataset, nb_samples: int, loss_save_file, model_save_file):
        x, y_true = next(iter(dataset.valid_data.batch(dataset.valid_data.cardinality())))
        y_samples, y_pred = self.model.predict(x, nb_samples)  # pass in the x value
        print(len(y_samples))
        print(len(y_pred))
        print(y_pred.shape)
        self.learning_diagnostics(loss_save_file, model_save_file)
        # Prediction Plot
        if dataset.likelihoodModel == "Regression":

            plt.figure(figsize=(10, 5))
            plt.scatter(range(len(y_true)), y_true, label='True Values', alpha=0.5)
            plt.scatter(range(len(y_pred)), y_pred, label='Predicted Mean', alpha=0.5)
            plt.legend()
            plt.title('True vs Predicted Values')
            plt.xlabel('Sample Index')
            plt.ylabel('Output')
            plt.show()

            self.metrics_regression(y_pred, y_true)
            err = np.sqrt(self.uncertainty_regression(y_samples))
            pred_dev = y_pred.numpy() - y_true.numpy()
            
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
            
        elif dataset.likelihoodModel == "Classification":
            y_pred_labels = tf.argmax(y_pred, axis=1)
            self.metrics_classification(y_pred, y_true)
            self.plot_2d_3d(x, y_true)
            skplt.metrics.plot_confusion_matrix(y_true, y_pred_labels, normalize=True, title = 'Confusion Matrix')
            plt.show()
            #skplt.metrics.plot_precision_recall(y_true, y_pred, title = 'Precision-Recall Curve')
            #plt.show()
            # self.uncertainty_classification(y_samples)
        else: 
            print("Invalid loss function")  
        
    def plot_2d_3d(self, x, y_true):
        # 2D visualisation
        x_2d = tf.convert_to_tensor(np.reshape(x, (x.shape[0], -1)), dtype=tf.dtypes.float32)
        eigenvalues, eigenvectors = tf.linalg.eigh(tf.tensordot(tf.transpose(x_2d), x_2d, axes=1))
        x_pca = tf.tensordot(x_2d, eigenvectors, axes=1)
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(x_pca[:, -1], x_pca[:, -2], c=y_true, s=5)
        legend_plt = ax.legend(*scatter.legend_elements(), loc="lower left", title="Digits")
        ax.add_artist(legend_plt)
        plt.title('First Two Dimensions of Projected Data After Applying PCA')
        plt.show()
        # 3D visualisation
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')
        plt_3d = ax.scatter3D(x_pca[:, -1], x_pca[:, -2], x_pca[:, -3], c=y_true, s=1)
        plt.colorbar(plt_3d)
        plt.title('First Three Dimensions of Projected Data After Applying PCA')
        plt.show()
            
    def metrics_regression(self, y_pred, y_true):
        mse = skmet.mean_squared_error(y_true, y_pred)
        rmse = skmet.mean_squared_error(y_true, y_pred, squared=False)
        mae = skmet.mean_absolute_error(y_true, y_pred)
        r2 = skmet.r2_score(y_true, y_pred)
        print("""Performence metrics for Regression:
              Mean Square Error: {}
              Root Mean Square Error: {}
              Mean Absolute Error: {}
              R^2: {}""".format(mse, rmse, mae, r2))
        
    def metrics_classification(self, y_pred, y_true):
        accuracy = skmet.accuracy_score(y_true, tf.argmax(y_pred, axis = 1))
        recall_score = skmet.recall_score(y_true, tf.argmax(y_pred, axis = 1), average= "macro")
        precision = skmet.precision_score(y_true, tf.argmax(y_pred, axis = 1), average= "micro")
        f1 = skmet.f1_score(y_true, tf.argmax(y_pred,axis = 1), average = "macro")
        print("""Performence metrics for Classification:
              Accuracy: {}
              Mean Recall: {}
              Mean Precision: {}
              F1-Score: {}""".format(accuracy, recall_score, precision, f1))
        
    def uncertainty_regression(self, y_samples) -> tuple:
        variance = np.var(y_samples, axis=0)
        return variance
        
    def uncertainty_classification(self, y_samples) -> tuple:
        # For classification, we might use the entropy of the predicted probabilities
        # as a measure of aleatoric uncertainty and variance of multiple stochastic
        # forward passes as epistemic uncertainty.

        # Assuming predict returns a distribution over classes for each sample
        mean = np.mean(y_samples, axis=0)
        variance = np.var(y_samples, axis=0)
        print("""Uncertainty for Classificaton: 
                Epistemic Uncertainty: {}
                Aleatoric Uncertainty: {}""".format(variance, mean))
        
    def learning_diagnostics(self, loss_file, model_file):
        if loss_file != None:
            losses = np.loadtxt(loss_file)
            plt.plot(losses)
            plt.title("Training Loss")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()