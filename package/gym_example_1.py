"""Interactive demo of the Gym environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.datasets.Dataset import Dataset
from src.dynamics.deep_pilco import BayesianDynamics, NNPolicy, DynamicsTraining
from src.optimizers.SWAG import SWAG
from src.optimizers.HyperParameters import HyperParameters

# Set up the environment
env = gym.make("LunarLander-v2", render_mode=None )
x_prev, info = env.reset(seed=42)

# x_array = []
# y_array = []


# # Run the game loop
# total_reward = 0
# for i in range(1000):
#     action = env.action_space.sample()  # this is where you would insert your policy
#     # policy.act(x_prev)
    
#     x, reward, terminated, truncated, info = env.step(action)
#     x_data = np.append(x_prev, action)
    
    
#     x_array.append(x_data)
#     y_array.append(x)
    
#     # total_reward += reward
    
#     if terminated or truncated:
#     #     xdata.append(i)
#     #     ydata.append(total_reward)
#     #     update_plot(xdata, ydata)  # Update the plot with the latest reward

#     #     total_reward = 0
#         x_prev, info = env.reset()  # Reset the environment
    
#     x_prev = x  # Update the previous state
    
#     # print(x_data)
#     # print(x)
#     # print(x_array)
#     # print(y_array)

# env.close()  # Close the environment when done

# convert to tf.data.Dataset
# dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array))


def state_reward(state):
    return sum(state)

# Neural network templates: only contain inner layers; no input/output layers
policy_nn = tf.keras.Sequential()
policy_nn.add(tf.keras.layers.Dense(32, activation='relu'))
policy_nn.add(tf.keras.layers.Dense(8, activation='relu'))

dyntrain_nn = tf.keras.Sequential()
dyntrain_nn.add(tf.keras.layers.Dense(64, activation='relu'))
dyntrain_nn.add(tf.keras.layers.Dense(16, activation='relu'))
hyperparams = HyperParameters(lr=1e-2, k=10, frequency=1, scale=1)

dyn_training = DynamicsTraining(SWAG(), [
    tf.keras.losses.MeanSquaredError(),"Regression", True],
    dyntrain_nn, 'relu', hyperparams)
nn_policy = NNPolicy(policy_nn, 'relu', hyperparams)


bnn = BayesianDynamics(
    env=env,
    horizon=20,
    dyn_training=dyn_training,
    policy=nn_policy,
    state_reward=state_reward,
    learn_config=(100, 30, 0.5),
)
dyn_training.compile_more(starting_model=dyn_training.model)

bnn.learn(nb_epochs=1000)