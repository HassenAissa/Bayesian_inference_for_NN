import sys
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src')
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src/optimizer')

from SWAG import SWAG, SwagHyperparam
import utils
import numpy as np
import tensorflow as tf



def test_swag_on_MNIST():
    xtrain, xtest, ytrain, ytest = utils.load_data(r"Bayesian_NN\git_version\Beyesian_inference_for_NN\test\dataset_HASYv2")
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    
    # normalize, add bias
    means = np.mean(xtrain, axis = 0, keepdims= True)
    stds = np.std(xtrain, axis = 0, keepdims= True)
    
    xtrain = utils.normalize_fn(xtrain, means, stds)
    xtest = utils.normalize_fn(xtest, means, stds)
    fraction_train = 0.1
    n_samples = xtrain.shape[0]
    rinds = np.random.permutation(n_samples)

    n_train = int(n_samples * fraction_train)
    xtest = xtrain[rinds[n_train:]]
    ytest = ytrain[rinds[n_train:]]

    xtrain = xtrain[rinds[:n_train]]
    ytrain = ytrain[rinds[:n_train]]
    n_classes = utils.get_n_classes(ytrain)
    xtrain = xtrain.reshape(xtrain.shape[0], 32, 32, 1)
    xtest = xtest.reshape(xtest.shape[0], 32, 32,1)


    base_model = tf.keras.Sequential()
    base_model.add(tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(32,32,1)))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu'))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
    base_model.add(tf.keras.layers.MaxPooling2D(2))
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(120, activation='relu'))
    base_model.add(tf.keras.layers.Dense(84, activation='relu'))
    base_model.add(tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax))


    hyperparameters = SwagHyperparam(
        k = 50,
        frequency = 2,
        scale=1,
        lr = 1e-2,
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    train_dataset = train_dataset.shuffle(train_dataset.cardinality()).batch(1).repeat(10000)



    swag_model = SWAG(
        base_model = base_model,
        dataloader = train_dataset,
        hyperparameters  = hyperparameters,
    )


    for _ in range(500):
        swag_model.step()


    pred_test = swag_model.base_model.predict_on_batch(xtest)
    compare = tf.equal(tf.math.argmax(pred_test, axis = -1), ytest)
    print(tf.math.argmax(pred_test, axis = -1))
    print(ytest)
    print(tf.reduce_sum(tf.cast(compare, tf.float32)/xtest.shape[0]))


    print("finished training")
    nb_samples = 3000


    test_dataset = tf.data.Dataset.from_tensor_slices((xtest, ytest))


    prediction = tf.zeros_like(xtest)
    distribution = swag_model.distribution()

    prediction = tf.zeros([xtest.shape[0], n_classes])
    for i in range(nb_samples):
        pred_labels = []

        pred = swag_model.predict(distribution, xtest)
        prediction += pred

    prediction /= nb_samples
    print(tf.math.argmax(prediction, axis = -1))
    compare = tf.equal(tf.math.argmax(prediction, axis = -1), ytest)
    print(tf.reduce_sum(tf.cast(compare, tf.float32)/xtest.shape[0]))
test_swag_on_MNIST()
