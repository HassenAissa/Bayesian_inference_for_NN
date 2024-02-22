"""Interactive demo of the Gym environment."""

import gymnasium as gym, time, pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PyAce.dynamics.deep_pilco import BayesianDynamics, NNPolicy, DynamicsTraining
from PyAce.optimizers import SWAG
from PyAce.optimizers import HyperParameters

def runner():
    # Set up the environment
    env = gym.make("Acrobot")
    x_prev, info = env.reset(seed=42)

    def state_reward(state, action, t):
        c1, c2, s1, s2 = state[0],state[2],state[1],state[3]
        height = -c1-(c1*c2-s1*s2)
        speed = pow(state[4] + state[5]/2, 2)
        return 100*(height*100+speed*37)

    print(">>Start learning")
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
        horizon=17,
        dyn_training=dyn_training,
        policy=nn_policy,
        state_reward=state_reward,
        learn_config=(64, 32, 0.7), # dynamic epochs, particle number, discount factor
    )
    dyn_training.compile_more(starting_model=dyn_training.model)

    bnn.learn(nb_epochs=5)


    # Run an interactive demo of the trained policy
    # Create the environment
    env = gym.make("Acrobot", render_mode="human")  # Use "human" render mode for visualization
    observation, info = env.reset(seed=42)

    total_reward = 0
    t = 0
    done = False

    # Run the game loop
    print(">>Start real game")
    plt.title("Accumulative reward over time step")
    ts = [0]
    rewards = [0]
    while not done:
        action = tf.reshape(bnn.policy.act(tf.convert_to_tensor(observation)), shape=bnn.action_d)
        # action = action.numpy()
        state, reward, terminated, truncated, info = bnn.env.step(tf.cast(action, tf.int32))
        total_reward += reward  # Accumulate the reward
        t += 1
        rewards.append(total_reward)
        ts.append(t)
        if terminated or truncated:
            done = True
        # You can add a delay here if the visualization is too fast
        time.sleep(0.05)

    print("--Game finished--")
    print(f"Total reward: {total_reward}")
    plt.plot(ts, rewards)
    plt.show()
    env.close()  # Close the environment when done

def test_srlz():
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

    f = open("dyn.pkl", "wb")
    pickle.dump(dyn_training, f)
    f.close()
    f = open("dyn.pkl", "rb")
    dyn_training:DynamicsTraining = pickle.load(f)
    print(dyn_training.out_activation)
