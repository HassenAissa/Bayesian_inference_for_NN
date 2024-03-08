import numpy as np
from Pyesian.nn import BayesianModel
from Pyesian.datasets import Dataset
import sklearn.metrics as skmet
import tensorflow as tf
import os
import sklearn as sk
import tensorflow_probability as tfp

class Metrics():
    """
        a class giving metrics for performance analysis of a model over a dataset
        Args:
            model (BayesianModel): trained model that will make the predictions
            dataset (Dataset): the dataset on which to calculate the metrics
    """
    def __init__(self, model: BayesianModel, dataset: Dataset):
        self._model: BayesianModel = model
        self._dataset: Dataset = dataset
        self._nb_predictions: int = 0
        self._cached_samples : list = None
        self._cached_prediction: tf.Tensor = None
        self._cached_true_values: tf.Tensor = None
        self._cached_input: tf.tensor = None

    
    def _get_predictions(self, input, n_boundaries, y_true):
        if (self._nb_predictions == n_boundaries 
            and y_true.shape == self._cached_true_values.shape):
            y_pred = self._cached_prediction
            if self._cached_prediction.shape[1] == 1 and self._dataset.likelihood_model == "Classification":
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1-self._cached_prediction, self._cached_prediction], axis=1)
            return self._cached_samples, y_pred, self._cached_true_values, self._cached_input
        else:
            y_samples, y_pred = self._model.predict(input, n_boundaries)  # pass in the x value
            self._nb_predictions = n_boundaries
            self._cached_input = input
            self._cached_samples = y_samples
            self._cached_prediction = y_pred
            self._cached_true_values = y_true
            if y_pred.shape[1] == 1 and self._dataset.likelihood_model == "Classification":
                # in the very specific case of binary classification with one neuron output convert it to two output
                y_pred = tf.stack([1-y_pred, y_pred], axis=1)
            return y_samples, y_pred, y_true, input
    
    def summary(self, n_boundaries: int = 30, n_samples: int = 100, data_type = "test", save_path = None):
        """
        Gives a summary for the most important metrics for regression and classification

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.
        """
    
        if self._dataset.likelihood_model == "Regression" :
            self.mse(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.rmse(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.mae(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.r2(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.log_likeliood(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)

            
        elif self._dataset.likelihood_model == "Classification":
            self.accuracy(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.recall(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.precision(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.f1_score(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.auroc(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)
            self.ece(n_boundaries= n_boundaries, n_samples= n_samples, data_type=data_type, save_path=save_path)

        else: 
            print("Invalid loss function")
            
            
        
    # Regression performance metrics
    
    def mse(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """
        Mean squared error.
        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.
        Raises:
            Exception: when this method is called on other than a regression dataset
        Returns:
            (float): the mean square error
        """
        if self._dataset.likelihood_model == "Classification":
            raise Exception("Mean squared error could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        res = skmet.mean_squared_error(y_true, y_pred)
        self._save(save_path, "MSE", res)
        print("MSE: {}".format(res))
        return res
    
    def rmse(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """Root Mean Squared Error

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a regression dataset

        Returns:
            (float): the root mean squared error
        """
        if self._dataset.likelihood_model == "Classification":
            raise Exception("Root mean squared error could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        res = skmet.root_mean_squared_error(y_true, y_pred)
        self._save(save_path, "RMSE", res)
        print("RMSE: {}".format(res))
        return res
    
    def mae(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """Mean Absolute Error

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a regression dataset

        Returns:
            (float): the mean absolute error
        """
        if self._dataset.likelihood_model == "Classification":
            raise Exception("Mean absolute error could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        res = skmet.mean_absolute_error(y_true, y_pred)
        self._save(save_path, "MAE", res)
        print("MAE: {}".format(res))
        return res
    
    def r2(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """R2 score

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a regression dataset

        Returns:
            (float): the R2 score
        """
        if self._dataset.likelihood_model == "Classification":
            raise Exception("R2 score could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        res = skmet.r2_score(y_true, y_pred)
        self._save(save_path, "R2", res)
        print("R2 score: {}".format(res))
        return res

    def log_likeliood(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """ Negative log likelihood

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a regression dataset

        Returns:
            (float): the negative loglikelihood
        """
        if self._dataset.likelihood_model == "Classification":
            raise Exception("Log likelihood could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        guassian_distribution = tfp.distributions.Normal(tf.cast(y_true, dtype = y_pred.dtype), tf.ones_like(y_true, dtype = y_pred.dtype))
        res = tf.reduce_mean(guassian_distribution.log_prob(y_pred))
        self._save(save_path, "log_likelihood", res)
        print("log likelihood: {}".format(res))
        return res   
            
        
    # Classification performance metrics
    
    def accuracy(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """Prediction accuracy

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a classification dataset

        Returns:
            (float): the prediction accuracy
        """
        if self._dataset.likelihood_model != "Classification":
            raise Exception("Log likelihood could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        if y_pred.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
        res = skmet.accuracy_score(y_true, tf.argmax(y_pred, axis = 1)) * 100
        self._save(save_path, "Accuracy", res)
        print("Accuracy: {}%".format(res))
        return res
    
    def precision(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """precision score = tp / (tp + fp)

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a classification dataset

        Returns:
            (float): the precision score
        """
        if self._dataset.likelihood_model != "Classification":
            raise Exception("Log likelihood could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        if y_pred.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
        res = skmet.recall_score(y_true, tf.argmax(y_pred, axis = 1), average= "macro") * 100
        self._save(save_path, "Precision", res)
        print("Precision: {}%".format(res))
        return res
    
    def recall(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """The recall = tp / (tp + fn)

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a classification dataset

        Returns:
            (float): the recall score
        """
        if self._dataset.likelihood_model != "Classification":
            raise Exception("Log likelihood could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        if y_pred.shape[1] == 1:
            # in the very specific case of binary classification with one neuron output convert it to two output
            y_pred = tf.stack([1 - y_pred, y_pred], axis=1)
        res = skmet.precision_score(y_true, tf.argmax(y_pred, axis = 1), average= "micro") * 100
        self._save(save_path, "Recall", res)
        print("Recall: {}%".format(res))
        return res
    
    def f1_score(self, n_boundaries: int = 30, n_samples = 100, data_type = "test", save_path = None):
        """F1 score

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.

        Raises:
            Exception: when this method is called on other than a classification dataset

        Returns:
            (float): the f1 score
        """
        if self._dataset.likelihood_model != "Classification":
            raise Exception("Log likelihood could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)

        res = skmet.f1_score(y_true, tf.argmax(y_pred,axis = 1), average = "macro")
        self._save(save_path, "F1_score", res)
        print("F1 score: {}".format(res))
        return res

    def ece(self, n_boundaries:int = 30, n_samples = 100, data_type = "test", save_path = None, n_bins = 5):
        """ Expected calibration error

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics, defaults to "test".
            save_path (string, optional): Path directory in which a report folder will be created and sotres the metric files. Defaults to None.
            n_bins (int): number of probabilty bins. Defaults to 5.

        Raises:
            Exception: when this method is called on other than a classification dataset

        Returns:
            (float): the expected calibration error
        """
        if self._dataset.likelihood_model != "Classification":
            raise Exception("Log likelihood could only be computed for regression")
        input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
        y_samples, y_pred, y_true, input = self._get_predictions(input, n_boundaries, y_true)
        res = tfp.stats.expected_calibration_error(n_bins, logits = y_pred, labels_true = y_true)
        self._save(save_path, "ECE", res)
        print("ECE: {}".format(res))
        return res
    
    def _get_x_y(self, n_samples=100, data_type="test"):
        tf_dataset = self._dataset.valid_data
        if data_type == "test":
            tf_dataset = self._dataset.test_data
        elif data_type == "train":
            tf_dataset = self._dataset.train_data
            
        x,y_true = next(iter(tf_dataset.batch(n_samples)))
        return x,y_true
        
    def classification_uncertainty(self,n_boundaries = 30, n_samples = 100, data_type = "test", save_path = None):
        
        if self._dataset.likelihood_model == "Classification":
            input, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
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
        
    def auroc(self, n_boundaries = 10, n_samples = 100, data_type = "test", save_path = None, multi_class = "ovr"):
        """ Area under receiver operating characteristic curve

        Args:
            n_boundaries (int): the number of sampled networks for the Monte Carlo approximation. Defaults to 30.
            nb_samples (int): number of samples from the dataset. Defaults to 100.
            data_type (string, optional): the split of the dataset on which to calculate the metrics. Defaults to "test".
            save_path (string, optional): Path to folder in which a report folder will be created and sotres the metrics. Defaults to None.
            n_bins (int): number of probabilty bins. Defaults to 5.
            multi_class ('ovr', 'ovo'): in the case of multiclass, caculate as one-vs-rest or one-vs-one for all classes. Defaults to "ovr".

        Raises:
            Exception: when this method is called on other than a classification dataset

        Returns:
            (float): the area under receiver operating characteristic curve
        """
        if self._dataset.likelihood_model != "Classification":
            raise ValueError("ROC can only be plotted for Classification")  
        x, y_true = self._get_x_y(n_samples = n_samples, data_type= data_type)
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

