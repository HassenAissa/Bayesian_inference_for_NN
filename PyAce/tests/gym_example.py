"""Interactive demo of the Gym environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from PyAce.datasets import Dataset
from PyAce.optimizers.hyperparameters import HyperParameters
from PyAce.dynamics import DeepPilco
from tensorflow.keras import models, layers
from PyAce.optimizers import SWAG
import time

def h2(state):
    return state[0]*state[2] - state[1]*state[3]

def ht_speed(state):
    height = 4-state[0] - state[0]*2 - h2(state)
    speed = pow(state[4], 2)
    return -(height)

def runner():
    policy = tf.keras.models.Sequential()
    policy.add(layers.Dense(10, activation='relu', input_shape=(6,)))
    policy.add(layers.Dense(3, activation=tf.keras.activations.softmax))
    policy_hyperparams = HyperParameters(lr = 1e-3)

    bayesian_model = tf.keras.models.Sequential()
    bayesian_model.add(layers.Dense(30, activation='relu', input_shape=(9,)))
    bayesian_model.add(layers.Dense(6, activation="linear"))
    bayesion_hyperparams = HyperParameters(batch_size= 10, lr = 1e-1, k = 30, scale = 1, frequency = 1)

    env = gym.make("Acrobot")
    deep_pilco = DeepPilco(
        env= env,
        policy= policy,
        policy_optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
        bayesian_model_json= bayesian_model.to_json(),
        bayesian_optimizer= SWAG,
        bayesian_hyperparameters= bayesion_hyperparams,
        n_optimizer_iterations= 1000,
        extra_parameters_for_optimizer= {"starting_model": bayesian_model},
        reward=ht_speed,
        horizon=10
    )

    deep_pilco.learn(10)

    env = gym.make("Acrobot", render_mode="human")  # Use "human" render mode for visualization
    observation, info = env.reset(seed=42)
    done = False
    rewards, total_reward = [], 0
    states = []
    actions = []
    while not done:
        states.append(observation)
        # normalize observation -1 to 1 when taking policy; 
        # action take is the actual action taken without normalization
        action_taken = policy.predict(tf.reshape(tf.convert_to_tensor(observation), (1,-1)))
        action_taken = tf.argmax(action_taken, axis = 1)
        actions.append(action_taken[0])
        observation, reward, terminated, truncated, info = env.step(action_taken[0].numpy())
        # reward = ht_speed(observation)
        total_reward += reward  # Accumulate the reward
        rewards.append(total_reward)
        if terminated or truncated:
            done = True
        # You can add a delay here if the visualization is too fast
        time.sleep(0.02)
    print(rewards)
    print(actions)
runner()