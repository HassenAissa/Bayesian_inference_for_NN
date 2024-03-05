"""Interactive demo of the Gym environment."""

from PyAce.distributions import GaussianPrior
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from PyAce.datasets import Dataset
from PyAce.optimizers.hyperparameters import HyperParameters
from PyAce.dynamics import DeepPilco
from tensorflow.keras import models, layers
from PyAce.optimizers import SWAG, BBB
import time



def reward_fn(state):
    res = (state[2]**2 + state[0]**2)
    return res


def runner():
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=.5)

    policy = tf.keras.models.Sequential()
    policy.add(layers.Dense(30, activation='relu', input_shape=(4,)))
    policy.add(layers.Dense(2, activation=tf.keras.activations.softmax))

    bayesian_model = tf.keras.models.Sequential()
    bayesian_model.add(layers.Dense(30, activation='relu', input_shape=(6,)))
    bayesian_model.add(layers.Dense(4, activation="linear"))
    bayesion_hyperparams = HyperParameters(batch_size= 30, lr = 1e-1, k = 5, scale = 1, frequency = 1)
    # bayesion_hyperparams = HyperParameters(lr=1e-2, alpha=0, batch_size=30)"prior": GaussianPrior(0.0,-3.0)
    env = gym.make("CartPole-v1")
    deep_pilco = DeepPilco(
        env= env,
        policy= policy,
        policy_optimizer= tf.keras.optimizers.SGD(learning_rate=1e-1),
        bayesian_model_json= bayesian_model.to_json(),
        bayesian_optimizer= SWAG,
        bayesian_hyperparameters= bayesion_hyperparams,
        n_optimizer_iterations= 100,
        extra_parameters_for_optimizer= {"starting_model": bayesian_model},
        reward=reward_fn,
        horizon = 50,
        k_particles=30,
        gamma=1.0
    )

    deep_pilco.learn(1000)

    env = gym.make('CartPole-v1', render_mode="human")  # Use "human" render mode for visualization
    observation, info = env.reset()
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
        print(reward)
        total_reward += reward  # Accumulate the reward
        rewards.append(total_reward)
        if terminated or truncated:
            done = True
        # You can add a delay here if the visualization is too fast
        time.sleep(0.02)
    print(rewards)
    print(actions)
runner()