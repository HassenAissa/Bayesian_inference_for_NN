"""Interactive demo of the Gym environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from PyAce.datasets import Dataset


def runner():
    # Set up the environment
    env = gym.make("LunarLander-v2", render_mode="human", )
    x_prev, info = env.reset(seed=42)


    # bayesian dynamics training
    """
    TODO:  set up the bayesian model and policy for gym env

    bnn = prior
    policy = prior

    D = create_dataset(env)
    episodes = 1000
    bnn.train(D, epochs=episodes)
    policy.optimize(D, bnn, epochs=episodes)

    reward = 0
    for i in range(episodes):
        x_t = env.reset()
        H = horizon
        for t in range(H):
            a_t = policy.act(x_t)
            x_t1, r_t, done, _ = env.step(a_t)
            D.append(x_t, a_t, r_t, x_t1)
            x_t = x_t1
            if done:
                break
    """


    # plt.ion()  # Turn on interactive mode for non-blocking plotting
    fig, ax = plt.subplots()
    xdata, ydata = [], []

    # Initialize the plot
    ax.set_xlim(0, 1000)
    ax.set_ylim(-500, 300)  # Adjust the y-axis limits according to expected rewards
    line, = ax.plot(xdata, ydata, 'r-', label='Reward per Episode')
    ax.set_title('Reward Graph')  # Title for the graph
    ax.set_xlabel('Episode')  # X-axis label
    ax.set_ylabel('Total Reward')  # Y-axis label
    ax.legend()  # Add a legend

    def update_plot(x, y):
        line.set_xdata(x)
        line.set_ydata(y)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()


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

    print(dataset)