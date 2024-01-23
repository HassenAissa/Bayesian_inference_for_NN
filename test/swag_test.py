import sys
import pytest
import tensorflow as tf

sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src/optimizer')

from SWAG import SWAG, SwagHyperparam
import numpy as np


def test_swag_on_distribution_succeed():

    base_model = tf.keras.Sequential()
    base_model.add(tf.keras.Input(shape=(20)))
    base_model.add(tf.keras.layers.Dense(8))
    base_model.add(tf.keras.layers.Dense(2))
    expected = tf.random.normal([1, 2])
    training_size = 10000
    training_input = []
    training_label = []

    for i in range(training_size):
        training_input.append(tf.random.normal([1,20]))
        training_label.append(expected)



    hyperparameters = SwagHyperparam(
        k = 10,
        frequency = 1,
        scale=1,
        lr = 1e-2,
        loss =  tf.keras.losses.MeanSquaredError()
    )
    epochs = 500
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.concat(training_input, axis = 0), tf.concat(training_label, axis = 0)))
    train_dataset = train_dataset.batch(64).repeat(epochs)


    swag_model = SWAG(
        base_model = base_model,
        dataloader = train_dataset,
        hyperparameters  = hyperparameters,
    )



    for i in range(epochs):
        if i%1000 == 0:
            print(i)
        swag_model.step()

    input = tf.random.normal([1, 20])
    print("finished training")

    print(expected)
    print(swag_model.base_model(input))

    sum = 0
    nb_samples = 3000
    distribution = swag_model.distribution()

    for i in range(nb_samples):
        out = swag_model.predict(distribution, input)
        print(out)
        sum += out
    sum = sum/nb_samples
    loss =tf.reduce_sum(((expected - sum) ** 2.0))
    print("loss ",loss)
    print("expected ", expected)
    print("prediction " , sum)

    print(5555555555555)
    print(expected)
    print(swag_model.base_model(input))

    # assert loss < 0.0005
    # if(loss < 0.0005):
    #     print("test successful")
    # else:
    #     print("test FAILED !!!!!!!")


test_swag_on_distribution_succeed()