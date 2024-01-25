from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class Dataset:
    testData: pd.DataFrame
    trainData: pd.DataFrame
    validData: pd.DataFrame
    likelihood: str
    scaler: MinMaxScaler

    def __init__(self, train=None, test=None, valid=None, likelihood=None):
        self.testData = test
        self.trainData = train
        self.validData = valid
        self.scaler = MinMaxScaler()

    def normalise(self):
        self.trainData = self.scaler.fit_transform(self.trainData)
        self.testData = self.scaler.transform(self.testData)
        self.validData = self.scaler.transform(self.validData)

def convert_sklearn_dataset(dataset, likelihood):
    train, test_valid = train_test_split(dataset["data"], test_size=0.2, shuffle=True)
    test, valid = train_test_split(test_valid, test_size=0.5, shuffle=True)
    new_dataset = Dataset(train=train, test=test, valid=valid, likelihood=likelihood)
    return new_dataset
