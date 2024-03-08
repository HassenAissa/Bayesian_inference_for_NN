import threading
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras import models, layers
import tensorflow as tf
from Pyesian.nn import BayesianModel
from Pyesian.datasets import Dataset
from Pyesian.distributions.GaussianPrior import GaussianPrior
from Pyesian.optimizers import HyperParameters
from Pyesian.optimizers import SWAG, BBB
from Pyesian.visualisations.Visualisation import Visualisation
from matplotlib.widgets import Button


def SWAG_test(dataset, base_model) -> BayesianModel:
    hyperparams = HyperParameters(lr=1e-2, k=100, frequency=1, scale=1)
    # instantiate your optimizer
    optimizer = SWAG()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(300)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def BBB_test(dataset, base_model) -> BayesianModel:
    # instantiate your optimizer
    hyperparams = HyperParameters(lr=1e-1, alpha=0.0, pi=0.2, batch_size=5000)

    # instantiate your optimizer
    optimizer = BBB()
    prior1 = GaussianPrior(0.0, -5.0)
    prior2 = GaussianPrior(0.0, 0.01)
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, prior=prior1, prior2=prior2)

    optimizer.train(300)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model


def create_base_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(50, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1, activation='linear')
    ])


# intervals of 1 to 10, 1/10, such as 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9...
x = tf.range(-2, 2, delta=0.001)
print(len(x))

y = 1.5*tf.cos(2*x)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression")

train_x_numpy = x.numpy()
train_y_numpy = y.numpy()

base_model = create_base_model()
bayesian_model: BayesianModel = BBB_test(dataset, base_model)

# now use dataset of x 1-10, 1/11
x = tf.range(-2, 2, delta=1/10)
x_numpy = x.numpy()

num_samples = 50
samples_results, final_result = bayesian_model.predict(
    x, nb_samples=num_samples)
print(final_result)
final_result = final_result.numpy()
# Plotting each Monte Carlo sample
fig, ax = plt.subplots()
plt.scatter(train_x_numpy, train_y_numpy, label='Original Data')
model_type = 'BBB'


def plot(initial=False):
    # Clear the current plot
    ax.cla()

    # Scatter plot of the training data
    ax.scatter(train_x_numpy, train_y_numpy, label='Training Data')

    # Create a range of x values for prediction
    x = tf.range(-2, 2, delta=1/10)
    x_numpy = x.numpy()

    # Get the predictions from the Bayesian model
    samples_results, final_result = bayesian_model.predict(x, 100)
    print(final_result)

    # Plot each Monte Carlo sample
    for i, sample_result in enumerate(samples_results):
        label = 'Monte Carlo samples' if i == 0 else None  # Label only the first line
        # ax.scatter(x_numpy, sample_result, color='red', s=10, alpha=0.1)
        ax.plot(x_numpy, sample_result, 'r-', alpha=0.1, label=label)

    # Plot the mean prediction
    ax.plot(x_numpy, final_result, 'b-', linewidth=2, label='Mean Prediction')

    # Set plot labels and title
    ax.set_title('Monte Carlo Predictions' if initial else 'Updated Monte Carlo Predictions')
    ax.set_xlabel('x')
    ax.set_ylabel('Predicted y')
    ax.legend()

    # Draw the updated plot
    fig.canvas.draw()





def select_and_train_model_in_thread(x, y):
    global model_type, bayesian_model

    # Retrain the model with the new data point
    # bayesian_model.train(x, y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = Dataset(
        dataset, tf.keras.losses.MeanSquaredError(), "Regression")

    # identify the model type
    if model_type == 'BBB':
        bayesian_model = BBB_test(dataset, base_model)
    elif model_type == 'SWAG':
        bayesian_model = SWAG_test(dataset, base_model)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    # Once training is complete, update the plot
    plot()


# Adjust the bottom to make more room for buttons
plt.subplots_adjust(bottom=0.3)
# Move the BBB button down
button_ax_bbb = fig.add_axes([0.25, 0.01, 0.1, 0.075])
# Move the SWAG button down
button_ax_swag = fig.add_axes([0.45, 0.01, 0.1, 0.075])
# Create the buttons
btn_bbb = Button(button_ax_bbb, 'BBB')
btn_swag = Button(button_ax_swag, 'SWAG')

# Define an event callback function to add a data point
def onclick(event):
    if button_ax_bbb.contains(event)[0] or button_ax_swag.contains(event)[0]:
        # Click is on a button, ignore it
        return
    # Check that the click was in the axes
    if event.inaxes is not None:
        global train_x_numpy, train_y_numpy, bayesian_model
        train_x_numpy = np.append(train_x_numpy, event.xdata)
        train_y_numpy = np.append(train_y_numpy, event.ydata)

        ax.scatter(event.xdata, event.ydata, color='green', s=50, alpha=0.6)
        fig.canvas.draw_idle()

        # Retrain and re-render the plot in a separate thread
        threading.Thread(target=select_and_train_model_in_thread, args=(
            train_x_numpy, train_y_numpy)).start()



fig.canvas.mpl_connect('button_press_event', onclick)


def train_and_update(model_type, dataset, base_model, ax):
    global bayesian_model
    if model_type == 'BBB':
        bayesian_model = BBB_test(dataset, base_model)
    elif model_type == 'SWAG':
        bayesian_model = SWAG_test(dataset, base_model)

    plot()


def on_button_clicked(event, model_type, dataset, base_model, ax):
    threading.Thread(target=train_and_update, args=(
        model_type, dataset, base_model, ax)).start()


def select_and_train_model(method, ax):
    global bayesian_model
    if method == 'BBB':
        bayesian_model = BBB_test(dataset=dataset, base_model=base_model)
    elif method == 'SWAG':
        bayesian_model = SWAG_test(dataset=dataset, base_model=base_model)
    plot()


def on_bbb_clicked(event):
    global model_type
    model_type = 'BBB'
    threading.Thread(target=select_and_train_model, args=('BBB', ax)).start()


def on_swag_clicked(event):
    global model_type
    model_type = 'SWAG'
    threading.Thread(target=select_and_train_model, args=('SWAG', ax)).start()


# Connect the callbacks
btn_bbb.on_clicked(on_bbb_clicked)
btn_swag.on_clicked(on_swag_clicked)


def main():
    plot(initial=True)
    plt.show()


if __name__ == '__main__':
    main()
