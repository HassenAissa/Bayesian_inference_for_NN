import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd




"""
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
"""

"""
class Dataset:
    def __init__(self, tf_dataset, loss):
        self._loss = loss
        self._tf_dataset = tf_dataset
    def loss(self):
        return self._loss
 
    def tf_dataset(self) -> tf.data.Dataset:
        return self._tf_dataset
"""
class Dataset:
    test_size: int
    valid_size: int
    train_size: int
    train_data: tf.data.Dataset
    test_data: tf.data.Dataset
    valid_data: tf.data.Dataset
    size: int

    def __init__(self, dataset, loss, likelihoodModel="Classification", normalise=True):
        self._loss = loss
        self.likelihoodModel = likelihoodModel
        if isinstance(dataset, str) and (dataset in tfds.list_builders()):
            dataset = tfds.load(dataset, split="train", try_gcs=True)
            dataset = dataset.map(lambda data: [data["image"], data["label"]])
            self._init_from_tf_dataset(dataset)
        elif (isinstance(dataset, tf.data.Dataset)):
            self._init_from_tf_dataset(dataset)
        elif(isinstance(dataset, pd.DataFrame)):
            self._init_from_dataframe(dataset) 
        elif(isinstance(dataset, str)):
            self._init_from_csv(dataset)
        else:
            raise Exception("Unsupported dataset format")
        self.train_data = self.train_data.shuffle(self.train_data.cardinality())
        if (normalise):
            self.normalise()

    # def get_likelihood_type(self):
    #     if isinstance(self._loss, tf.keras.losses.MeanSquaredError):
    #         return "Regression"
    #     elif isinstance(self._loss, tf.keras.losses.SparseCategoricalCrossentropy):
    #         return "Classification"
    #     else:
    #         return None
        
    #this is just to be in line with old api
    def _init_from_tf_dataset(self, dataset: tf.data.Dataset):
        self.size = tf.data.experimental.cardinality(dataset).numpy()
        self.train_size = int(0.8 * self.size)
        self.test_size = int(0.1 * self.size)
        self.valid_size = int(0.1 * self.size)
        self.train_data = dataset.take(self.train_size)
        self.test_data = dataset.skip(self.train_size)
        self.valid_data = self.test_data.skip(self.test_size)
        self.test_data = self.test_data.take(self.test_size)

    def _init_from_dataframe(self, dataframe: pd.DataFrame):
        targets = dataframe.pop('target')
        dataset = tf.data.Dataset.from_tensor_slices((dataframe.values, targets.values))
        self._init_from_tf_dataset(dataset)

    def _init_from_csv(self, filename: str):
        dataframe = pd.read_csv(filename)
        self._init_from_dataframe(dataframe)

    def tf_dataset(self) -> tf.data.Dataset:
        return self.train_data
    
    def loss(self):
        return self._loss

    def map(self, x,y, mean, std, label_mean, label_std):
        return ((x-mean)/std, (y-label_mean)/label_std)
    
    def normalise(self):
        input, label = next(iter(self.train_data.batch(self.train_data.cardinality().numpy()/10)))
        input = tf.reshape(input, (-1,))
        mean = tf.reduce_mean(input)
        std = tf.math.reduce_std(tf.cast(input, dtype = tf.float32))
        label_mean = tf.reduce_mean(label)
        label_std = tf.math.reduce_std(tf.cast(label, dtype = tf.float32))
        #TODO: do we normalize the labels for regression???????

        self.train_data = self.train_data.map(lambda x,y: self.map(x,y,mean,std+1e-8, label_mean, label_std+1e-8))
        self.valid_data = self.valid_data.map(lambda x,y: self.map(x,y,mean,std+1e-8, label_mean, label_std+1e-8))
        self.test_data = self.test_data.map(lambda x,y: self.map(x,y,mean,std+1e-8, label_mean, label_std+1e-8))


        


def load_from_tf_dataset(dataset_name, likelihoodModel: str) -> Dataset:
    data = tfds.load(dataset_name, split='train', shuffle_files=True)
    assert isinstance(data, tf.data.Dataset)
    loss = tf.keras.losses.MeanSquaredError() if likelihoodModel == "Classification" else tf.keras.losses.SparseCategoricalCrossentropy()
    return Dataset(data, loss, likelihoodModel)
    
"""
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
"""
"""
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
"""

# load_tf_dataset('mnist')
"""
data = tfds.load('mnist', split='train')
assert isinstance(data, tf.data.Dataset)
print(data.take(10))
print(data.__dict__)
dataset = convert_tf_dataset(data, 100)
print(dataset.train_data)"""