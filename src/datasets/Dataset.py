from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd


class Dataset:
    testData: dict
    trainData: dict
    validData: dict
    likelihoodModel: str
    scaler: MinMaxScaler

    def __init__(self, train=None, test=None, valid=None, likelihood=None):
        self.testData = test
        self.trainData = train
        self.validData = valid
        self.likelihoodModel = likelihood
        self.scaler = MinMaxScaler()

    def normalise(self):
        self.trainData["input"] = self.scaler.fit_transform(self.trainData["input"])
        self.testData["input"] = self.scaler.transform(self.testData["input"])
        self.validData["input"] = self.scaler.transform(self.validData["input"])

def convert_sklearn_dataset(dataset, likelihood):
    df = dataset["data"].insert(4, "target", dataset["target"])
    print(df)
    train, test_valid = train_test_split(df, test_size=0.2, shuffle=True)
    test, valid = train_test_split(test_valid, test_size=0.5, shuffle=True)
    train_data = {"input": None, "labels": None}
    test_data = {"input": None, "labels": None}
    valid_data = {"input": None, "labels": None}
    train_data["labels"] = tf.convert_to_tensor(train["target"])
    test_data["labels"] = tf.convert_to_tensor(test["target"])
    valid_data["labels"] = tf.convert_to_tensor(valid["target"])
    train_data["input"] = tf.convert_to_tensor(train.drop("target"))
    test_data["input"] = tf.convert_to_tensor(test.drop("target"))
    valid_data["input"] = tf.convert_to_tensor(valid.drop("target"))
    new_dataset = Dataset(train=train_data, test=test_data, valid=test_data, likelihood=likelihood)
    return new_dataset

def convert_csv_dataset(filename, likelihood):
    df = pd.read_csv(filename)
    train, test_valid = train_test_split(df, test_size=0.2, shuffle=True)
    test, valid = train_test_split(test_valid, test_size=0.5, shuffle=True)
    train_data = {"input": None, "labels": None}
    test_data = {"input": None, "labels": None}
    valid_data = {"input": None, "labels": None}
    train_data["labels"] = tf.convert_to_tensor(train["target"])
    test_data["labels"] = tf.convert_to_tensor(test["target"])
    valid_data["labels"] = tf.convert_to_tensor(valid["target"])
    train_data["input"] = tf.convert_to_tensor(train.drop("target"))
    test_data["input"] = tf.convert_to_tensor(test.drop("target"))
    valid_data["input"] = tf.convert_to_tensor(valid.drop("target"))
    new_dataset = Dataset(train=train_data, test=test_data, valid=test_data, likelihood=likelihood)
    return new_dataset


data = convert_sklearn_dataset(load_iris(as_frame=True), "classification")
data.normalise()

print(load_iris())
print(data.testData)