"""Interactive demo of the Gym environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from src.datasets.Dataset import Dataset
from src.dynamics.deep_pilco import BayesianDynamics, Policy, NNPolicy
from src.optimizers.SWAG import SWAG
from src.optimizers.HyperParameters import HyperParameters

# Set up the environment
env = gym.make("LunarLander-v2", render_mode="human", )
x_prev, info = env.reset(seed=42)



x_array = []
y_array = []


# Run the game loop
total_reward = 0
for i in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    # policy.act(x_prev)
    
    x, reward, terminated, truncated, info = env.step(action)
    x_data = np.append(x_prev, action)
    
    
    x_array.append(x_data)
    y_array.append(x)
    
    # total_reward += reward
    
    if terminated or truncated:
    #     xdata.append(i)
    #     ydata.append(total_reward)
    #     update_plot(xdata, ydata)  # Update the plot with the latest reward

    #     total_reward = 0
        x_prev, info = env.reset()  # Reset the environment
    
    x_prev = x  # Update the previous state
    
    # print(x_data)
    # print(x)
    # print(x_array)
    # print(y_array)

env.close()  # Close the environment when done

# convert to tf.data.Dataset
dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_array, y_array))



nn_policy = NNPolicy(
    network=tf.keras.Sequential([
        tf.keras.layers.Dense(84, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ]),
    hyperparams={"lr": 1e-2}
)

def state_reward(state):
    return 1
  
  
base_model = tf.keras.Sequential()
base_model.add(tf.keras.layers.Dense(84, activation='linear', input_shape=(8,) ))

hyperparams = HyperParameters(lr=1e-2, k=10, frequency=1, scale=1)
optimizer = SWAG()


bnn = BayesianDynamics(
    env=env,
    n_episodes=100,
    policy=nn_policy,
    state_reward=state_reward,
    dyntrain_config=[optimizer, hyperparams, base_model, None, tf.keras.losses.SparseCategoricalCrossentropy(), "Classification", True],
    learn_config=(100, 50, .5),
)

bnn.learn(nb_epochs=1000)