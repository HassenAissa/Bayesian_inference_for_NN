import threading
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras import models, layers
import tensorflow as tf
from PyAce.nn import BayesianModel
from PyAce.datasets import Dataset
from PyAce.optimizers import HyperParameters
from PyAce.optimizers import SWAG
from PyAce.visualisations.Visualisation import Visualisation

# intervals of 1 to 10, 1/10, such as 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9...
x = tf.range(-2, 2, delta=1/20)
print(len(x))

# cosine of x 1.5cos2x + 2.3
y = 0.5*tf.cos(2*x)
dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression")

train_x_numpy = x.numpy()
train_y_numpy = y.numpy()

# Plot the data



def SWAG_test(dataset, base_model) -> BayesianModel:
    hyperparams = HyperParameters(lr=1e-2, k=100, frequency=1, scale=1)
    # instantiate your optimizer
    optimizer = SWAG()
    optimizer.compile(hyperparams, base_model.to_json(),
                      dataset, starting_model=base_model)
    optimizer.train(1000)
    bayesian_model: BayesianModel = optimizer.result()
    return bayesian_model





base_model = tf.keras.models.Sequential(
    [
        # More neurons and tanh activation
        tf.keras.layers.Dense(20, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(30, activation='relu'),  # Additional layer
        tf.keras.layers.Dense(20, activation='relu'),  # Additional layer
        tf.keras.layers.Dense(10, activation='relu'),  # Additional layer
        tf.keras.layers.Dense(1, activation='linear')  # Output layer
    ]
)
bayesian_model: BayesianModel = SWAG_test(dataset, base_model)




# now use dataset of x 1-10, 1/11
x = tf.range(-2, 2, delta=1/10)
x_numpy = x.numpy()

samples_results, final_result = bayesian_model.predict(x, 100)
print(final_result)
final_result = final_result.numpy()
# Plotting each Monte Carlo sample

fig, ax = plt.subplots()

plt.scatter(train_x_numpy, train_y_numpy)
# plt.title('Plot of 0.5*cos(2x)')


def retrain_and_plot():
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
    for sample_result in samples_results:
        ax.scatter(x_numpy, sample_result, color='red', s=10, alpha=0.1)
        ax.plot(x_numpy, sample_result, 'r-', alpha=0.1)

    # Plot the mean prediction
    ax.plot(x_numpy, final_result, 'b-', linewidth=2)  # blue line, thicker

    # Set plot labels and title
    ax.set_title('Updated Monte Carlo Predictions')
    ax.set_xlabel('x')
    ax.set_ylabel('Predicted y')
    ax.legend()

    # Draw the updated plot
    fig.canvas.draw()

for sample_result in samples_results:
    sample_result = sample_result.numpy()
    plt.scatter(x_numpy, sample_result, color='red', s=10)
    plt.plot(x_numpy, sample_result, 'r-', alpha=0.1)

# Plot the mean prediction
plt.plot(x_numpy, final_result, 'b-', linewidth=2)  # blue line, thicker
plt.scatter(x_numpy,  final_result, color='blue', s=10)

plt.title('Monte Carlo Predictions')
plt.xlabel('x')
plt.ylabel('Predicted y')



def train_model_in_thread(x, y):
    global bayesian_model
    # Retrain the model with the new data point
    # bayesian_model.train(x, y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = Dataset(dataset, tf.keras.losses.MeanSquaredError(), "Regression")
    bayesian_model = SWAG_test(dataset, base_model)
    
    # Once training is complete, update the plot
    retrain_and_plot()
    
# Define an event callback function to add a data point
def onclick(event):
    # Check that the click was in the axes
    if event.inaxes is not None:
        
        # fig.canvas.draw()  # Redraw the figure to show the new data point
        
        global train_x_numpy, train_y_numpy, bayesian_model
        train_x_numpy = np.append(train_x_numpy, event.xdata)
        train_y_numpy = np.append(train_y_numpy, event.ydata)
        
        ax.scatter(event.xdata, event.ydata, color='green', s=50, alpha=0.6)
        fig.canvas.draw_idle()
        

        # Retrain and re-render the plot
        # retrain_and_plot()
        
        threading.Thread(target=train_model_in_thread, args=(train_x_numpy, train_y_numpy)).start()

# Connect the event to the callback function
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()