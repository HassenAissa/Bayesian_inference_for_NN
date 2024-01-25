import numpy as np
from src.nn.BayesianModel import BayesianModel
from pydantic import BaseModel
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, recall_score, precision_score, accuracy_score, roc_auc_score
import sklearn.metrics as met
import tensorflow as tf

""""imaginary dataset class"""

class Dataset(BaseModel):
    likelihood: str # "Regressor" or "Classification"
    output: float
    testData: tf.Tensor

class Analytics(BayesianModel):
    def __init__(self):
        self.conf_intv = 0.2
        pass
    
    def metrics(self, dataset):
        y_pred = self.predict(dataset.testData.input)  # pass in the x value
        y_true = dataset.testData.labels # true value
        if dataset.likelihoodModel == "Regressor":
            return self.metrics_regressor(y_pred, y_true)
        else:
            return self.metrics_classification(y_pred, y_true)
        
    # https://seaborn.pydata.org
    def visualise():
        # ouptut results with table
        # plot y_true VS y_pred
        pass
        
    def metrics_regressor(self, y_pred, y_true):
        mse = met.mean_squared_error(y_true, y_pred)
        rmse = met.mean_squared_error(y_true, y_pred, squared=False)
        mae = met.mean_absolute_error(y_true, y_pred)
        r2 = met.r2_score(y_true, y_pred)
        print("Performence metrics for Regression: ....")
        
    def metrics_classification(self, y_pred, y_true):
        accuracy = met.accuracy_score(y_true, y_pred)
        recall_score = met.recall_score(y_true, y_pred)
        precision = met.precision_score(y_true, y_pred)
        f1 = met.f1_score(y_true, y_pred)
        print("Performence metrics for Classification: ....")
        
    def uncertainty(self, dataset, n_samples=100) -> tuple:
        # epistemic - model's lack of knowledge
        # aleatoric - natural variability or noise in the data
        if dataset.likelihoodModel == "Regressor":
            return self.uncertainty_regressor(dataset, n_samples)
        else:
            return self.uncertainty_classification(dataset, n_samples)
        
    
    def uncertainty_regressor(self, dataset, n_samples=100) -> tuple:
        predictions = [self.predict(dataset.testData.input) for _ in range(n_samples)]
        # means = np.mean(predictions, axis=0)
        # variance = np.var(predictions, axis=0)
        
    def uncertainty_classification(self, dataset, n_samples=100) -> tuple:
        # For classification, we might use the entropy of the predicted probabilities
        # as a measure of aleatoric uncertainty and variance of multiple stochastic
        # forward passes as epistemic uncertainty.

        # Assuming predict returns a distribution over classes for each sample
        predictions = self.predict(dataset.testData.input) # shape: (n_samples, n_classes)
        # predictions = np.array([self.predict(dataset.testData.input) for _ in range(n_samples)])
        samples = predictions
        aleatoric_uncertainty = np.mean(samples, axis=0)
        
        #aleatoric_uncertainty = -np.sum(mean_predictions * np.log(mean_predictions + 1e-10), axis=1) # Aleatoric Uncertainty as entropy
        epistemic_uncertainty = np.var(samples, axis=0) # Epistemic Uncertainty as variance of the predictions
        return aleatoric_uncertainty, epistemic_uncertainty
        
        
    def learning_diagnostics():
        pass