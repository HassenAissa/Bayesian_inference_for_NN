import numpy as np
from src.nn.BayesianModel import BayesianModel
from src.datasets.Dataset import Dataset
import sklearn.metrics as met
import tensorflow as tf

class Visualisation(BayesianModel):
    def __init__(self):
        pass
    
    # https://seaborn.pydata.org
    def visualise(self, dataset: Dataset, nb_samples: int):
        valid_dataset = iter(dataset.valid)
        x, y_true = next(valid_dataset)
        y_samples, y_pred = self.predict(x, nb_samples)  # pass in the x value
        if dataset.likelihoodModel == "Regressor":
            self.metrics_regressor(y_pred, y_true)
            self.uncertainty_regressor(y_samples)
        else:
            self.metrics_classification(y_pred, y_true)
            self.uncertainty_classification(y_samples)
        
    def metrics_regressor(self, y_pred, y_true):
        mse = met.mean_squared_error(y_true, y_pred)
        rmse = met.mean_squared_error(y_true, y_pred, squared=False)
        mae = met.mean_absolute_error(y_true, y_pred)
        r2 = met.r2_score(y_true, y_pred)
        print("""Performence metrics for Regression:
              Mean Square Error: {}
              Root Mean Square Error: {}
              Mean Absolute Error: {}
              R^2: {}""".format(mse, rmse, mae, r2))
        
    def metrics_classification(self, y_pred, y_true):
        accuracy = met.accuracy_score(y_true, y_pred)
        recall_score = met.recall_score(y_true, y_pred)
        precision = met.precision_score(y_true, y_pred)
        f1 = met.f1_score(y_true, y_pred)
        print("""Performence metrics for Classification:
              Accuracy: {}
              Mean Recall: {}
              Mean Precision: {}
              F1-Score: {}""".format(accuracy, recall_score, precision, f1))
        
    def uncertainty_regressor(self, y_samples) -> tuple:
        variance = np.var(y_samples, axis=0)
        print("""Uncertainty for Regression: 
              Epistemic Uncertainty: {}""".format(variance))
        
    def uncertainty_classification(self, y_samples) -> tuple:
        # For classification, we might use the entropy of the predicted probabilities
        # as a measure of aleatoric uncertainty and variance of multiple stochastic
        # forward passes as epistemic uncertainty.

        # Assuming predict returns a distribution over classes for each sample
        mean = np.mean(y_samples, axis=0)
        variance = np.var(y_samples, axis=0)
        print("""Uncertainty for Regression: 
                Epistemic Uncertainty: {}
                Aleatoric Uncertainty: {}""".format(variance, mean))
        
    def learning_diagnostics():
        pass