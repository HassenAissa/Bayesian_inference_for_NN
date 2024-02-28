import numpy as np
from PyAce.nn import BayesianModel
from PyAce.datasets import Dataset
import sklearn.metrics as skmet
import tensorflow as tf
import os
import sklearn as sk


class Metrics():
    """
        a class representing the performance analysis of a model
    """
    def __init__(self, model, dataset: Dataset):
        self._model: BayesianModel = model
        self._dataset: Dataset = dataset
        self._nb_predictions: int = 0
        self._cached_samples : list = None
        self._cached_prediction: tf.Tensor = None
        self._cached_true_values: tf.Tensor = None
        self._cached_input: tf.tensor = None

    
    def _get_predictions(self, input, nb_boundaries, y_true):
        if (self._nb_predictions == nb_boundaries 
            and y_true.shape == self._cached_true_values.shape):
            y_pred = self._cached_prediction
            if self._cached_prediction.shape[1] == 1 and self._dataset.likelihood_model == "Classification":
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1-self._cached_prediction, self._cached_prediction], axis=1)
            return self._cached_samples, y_pred, self._cached_true_values, self._cached_input
        else:
            y_samples, y_pred = self._model.predict(input, nb_boundaries)  # pass in the x value
            self._nb_predictions = nb_boundaries
            self._cached_input = input
            self._cached_samples = y_samples
            self._cached_prediction = y_pred
            self._cached_true_values = y_true
            if y_pred.shape[1] == 1 and self._dataset.likelihood_model == "Classification":
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1-y_pred, y_pred], axis=1)
            return y_samples, y_pred, y_true, input
    
    def summary(self, nb_boundaries: int, save_path = None):
        """
        outputs visualisations of performance metrics, learning diagnostic and uncertainty calculated upon the testing sub-dataset of given dataset. 

        Args:
            dataset (Dataset): dataset to perform analysis upon. Will use the testing sub-dataset.
            nb_samples (int): number of samples
            save_path (_type_, optional): Path to file storing metrics. Defaults to None.
        """
    
        if self._dataset.likelihood_model == "Regression" :
            self.mse(nb_boundaries, save_path)
            self.rmse(nb_boundaries, save_path)
            self.mae(nb_boundaries, save_path)
            self.r2(nb_boundaries, save_path)

            
        elif self._dataset.likelihood_model == "Classification":
            self.accuracy(nb_boundaries, save_path)
            self.recall(nb_boundaries, save_path)
            self.precision(nb_boundaries, save_path)
            self.f1_score(nb_boundaries, save_path)
            self.auroc()

        else: 
            print("Invalid loss function")
            
            
        
    # Regression performance metrics
    
    def mse(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        res = skmet.mean_squared_error(y_true, y_pred)
        self._save(save_path, "MSE", res)
        print("MSE: {}".format(res))
        return res
    
    def rmse(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        res = skmet.root_mean_squared_error(y_true, y_pred)
        self._save(save_path, "RMSE", res)
        print("RMSE: {}".format(res))
        return res
    
    def mae(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        res = skmet.mean_absolute_error(y_true, y_pred)
        self._save(save_path, "MAE", res)
        print("MAE: {}".format(res))
        return res
    
    def r2(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        res = skmet.r2_score(y_true, y_pred)
        self._save(save_path, "R2", res)
        print("R2 score: {}".format(res))
        return res

    # def log_likeliood(self, nb_boundaries: int, save_path = None):
    #     input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
    #     y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
    #     self._save(save_path, "log_likelihood", res)
    #     print("log likelihood: {}".format(res))
    #     return res   
            
        
    # Classification performance metrics
    
    def accuracy(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        if y_pred.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
        res = skmet.accuracy_score(y_true, tf.argmax(y_pred, axis = 1)) * 100
        self._save(save_path, "Accuracy", res)
        print("Accuracy: {}%".format(res))
        return res
    
    def precision(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        if y_pred.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
        res = skmet.recall_score(y_true, tf.argmax(y_pred, axis = 1), average= "macro") * 100
        self._save(save_path, "Precision", res)
        print("Precision: {}%".format(res))
        return res
    
    def recall(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)
        if y_pred.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
        res = skmet.precision_score(y_true, tf.argmax(y_pred, axis = 1), average= "micro") * 100
        self._save(save_path, "Recall", res)
        print("Recall: {}%".format(res))
        return res
    
    def f1_score(self, nb_boundaries: int, save_path = None):
        input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, input = self._get_predictions(input, nb_boundaries, y_true)

        res = skmet.f1_score(y_true, tf.argmax(y_pred,axis = 1), average = "macro")
        self._save(save_path, "F1_score", res)
        print("F1 score: {}%".format(res))
        return res
    
    def _get_x_y(self, n_samples=100, data_type="test"):
        tf_dataset = self._dataset.valid_data
        if data_type == "test":
            tf_dataset = self._dataset.test_data
        elif data_type == "train":
            tf_dataset = self._dataset.train_data
        x,y_true = next(iter(tf_dataset.batch(n_samples)))
        return x,y_true
        
    def classification_uncertainty(self,n_samples = 100, data_type = "test", n_boundaries = 30):
        if self._dataset.likelihood_model == "Classification":
            input, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
            y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
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
        
    def auroc(self, n_boundaries = 10, save_path = None, multi_class = "ovr"):
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("ROC can only be plotted for Classification")  
        x, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
        y_samples, y_pred, y_true, x = self._get_predictions(x, n_boundaries, y_true)
        y_pred = tf.reshape(y_pred, y_pred.shape[:2])
        one_hot_y_true = tf.one_hot(y_true, y_pred.shape[1])
        res = sk.metrics.roc_auc_score(one_hot_y_true, y_pred, average = "micro", multi_class = multi_class)
        self._save(save_path, "AUROC", res)
        print("AUROC: {}".format(res))
        return res
        
    def _save(self, save_path, name, content):
        if save_path != None:
            directory = os.path.join(save_path,"report")
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, name)
            f = open(file_path, "w")
            f.write(str(content))
            f.close()

