import numpy as np
from PyAce.nn import BayesianModel
from PyAce.datasets import Dataset
import sklearn.metrics as skmet
import tensorflow as tf


class Metrics():
    """
        a class representing the performance analysis of a model
    """
    def __init__(self, model, dataset: Dataset):
        self._model = model
        self._dataset = dataset
    
    def summary(self, nb_boundaries: int, loss_save_file = None):
        """
        outputs visualisations of performance metrics, learning diagnostic and uncertainty calculated upon the testing sub-dataset of given dataset. 

        Args:
            dataset (Dataset): dataset to perform analysis upon. Will use the testing sub-dataset.
            nb_samples (int): number of samples
            loss_save_file (_type_, optional): Path to file storing loss values throughout training. Defaults to None.
        """
        x, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred = self._model.predict(x, nb_boundaries)  # pass in the x value

        if self._dataset.likelihood_model == "Regression" :
            print("MSE:", self.mse(y_pred, y_true))
            print("RMSE:", self.rmse(y_pred, y_true))
            print("MAE:", self.mae(y_pred, y_true))
            print("R2:", self.r2(y_pred, y_true))
            
        elif self._dataset.likelihood_model == "Classification":
            if y_pred.shape[1] == 1:
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
            print("Accuracy: {}%".format(self.accuracy(y_pred, y_true)))
            print("Recall: {}%".format(self.recall(y_pred, y_true)))
            print("Precision: {}%".format(self.precision(y_pred, y_true)))
            print("F1 Score: {}%".format(self.f1_score(y_pred, y_true)))
            print("ROC Score: {}".format(self.auroc(y_pred, y_true)))
        else: 
            print("Invalid loss function")
            
            
        
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
    
    def auroc(self, y_pred, y_true):
        y_pred = tf.argmax(y_pred, axis = 1)
        roc = skmet.roc_auc_score(y_true, y_pred, average='macro', multi_class="ovr")
        return roc

    def _get_x_y(self, n_samples=100, data_type="test"):
        tf_dataset = self._dataset.valid_data
        if data_type == "test":
            tf_dataset = self._dataset.test_data
        elif data_type == "train":
            tf_dataset = self._dataset.train_data
        x,y_true = next(iter(tf_dataset.batch(n_samples)))
        return x,y_true
        
    def uncertainty(self,n_samples = 100, data_type = "test", n_boundaries = 30):
        if self._dataset.likelihood_model == "Classification":
            x,y_true = self._get_x_y(n_samples, data_type)
            y_samples, y_pred = self._model.predict(x, n_boundaries)
            if y_pred.shape[1] == 1:
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
            aleatorics = 0
            epistemics = 0
            for sample in y_samples:
                aleatoric = 0
                epistemic = 0
                nb_classes = y_pred.shape[1]
                if sample.shape[1] == 1:
                    # in the very specific case of binary classification with one neuron output convert it to two output
                    sample = tf.stack([1 - sample, sample], axis=1)
                aleatorics_tmp = []
                epistemics_tmp = []
                for prediction, label in zip(sample, y_true):
                    prediction_as_1d_matrix = tf.reshape(prediction, (-1,1))
                    aleatoric += (tf.linalg.diag(prediction) 
                                  - tf.matmul(prediction_as_1d_matrix, tf.transpose(prediction_as_1d_matrix)))
                    epistemic_deviation = prediction_as_1d_matrix - tf.one_hot(label, nb_classes)
                    epistemic += tf.matmul(epistemic_deviation, tf.transpose(epistemic_deviation))
                    epistemics_tmp.append(epistemic)
                    aleatorics_tmp.append(aleatoric)
                aleatorics += np.asarray(aleatorics_tmp)
                epistemics += np.asarray(epistemics_tmp)

            epistemics /= n_samples
            aleatorics /= n_samples
            return epistemics + aleatorics, aleatorics, epistemics
        else:
            raise Exception("only for classification")

