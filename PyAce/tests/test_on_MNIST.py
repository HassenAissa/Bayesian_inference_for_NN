import tensorflow as tf
from PyAce.datasets import Dataset
from PyAce.nn import BayesianModel
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations import Visualisation
from PyAce.datasets.utils import imgdata_preprocess, get_n_classes


def runner():
    x,y = imgdata_preprocess(r"...", 0.1, (32,32,1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    n_classes = get_n_classes(y)

    dataset = Dataset(
        dataset,
        tf.keras.losses.SparseCategoricalCrossentropy(),
        "Classification"
    )




    base_model = tf.keras.Sequential()

    base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32, 32, 1)))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(120, activation='relu'))
    base_model.add(tf.keras.layers.Dense(84, activation='relu'))
    base_model.add(tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax))

    hyperparams = HyperParameters(lr=1e-2, k=10, frequency=1, scale=1)
    # instantiate your optimizer
    optimizer = SWAG()
    # compile the optimizer with your data
    # this is a specification of SWAG, SWAG needs a starting_model from which to start the gradient descend
    optimizer.compile(hyperparams, base_model.to_json(), dataset, starting_model=base_model)
    optimizer.train(1000)

    bayesian_model: BayesianModel = optimizer.result()

    _, prediction = bayesian_model.predict(x, 100)
    compare = tf.equal(tf.math.argmax(prediction, axis=-1), y)
    print("Accuracy of the bayesian model :", tf.reduce_sum(tf.cast(compare, tf.float32) / x.shape[0]).numpy())
    # path = r"..."

    # bayesian_model.store(path)
    # bayesian_model: BayesianModel= BayesianModel.load(path)

    _, prediction = bayesian_model.predict(x, 100)

    compare = tf.equal(tf.math.argmax(prediction, axis=-1), y)

    print("Accuracy of the bayesian model after store/load :", tf.reduce_sum(tf.cast(compare, tf.float32) / x.shape[0]).numpy())

    analytics_builder = Visualisation(bayesian_model)

    analytics_builder.visualise(dataset, 100)