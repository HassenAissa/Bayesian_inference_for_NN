"""Interactive demo of the Gym environment."""

import gymnasium as gym, time, pickle, json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PyAce.dynamics.deep_pilco import BayesianDynamics, NNPolicy, DynamicsTraining
from PyAce.optimizers import SWAG
from PyAce.optimizers.hyperparameters import HyperParameters

def runner():
    
    print(">>Start learning")
    # Neural network templates: only contain inner layers; no input/output layers
    policy_template = tf.keras.Sequential()
    policy_template.add(
        tf.keras.layers.experimental.RandomFourierFeatures(output_dim=50)
    ) 
    pol_hyp = HyperParameters(lr=1e-2,batch_size=10)
    policy = NNPolicy(policy_template, pol_hyp)

    dyntrain_nn = tf.keras.Sequential()
    dyntrain_nn.add(tf.keras.layers.Dense(64, activation='sigmoid'))
    dyntrain_nn.add(tf.keras.layers.Dense(256, activation='tanh'))
    dyntrain_nn.add(tf.keras.layers.Dense(64, activation='sigmoid'))
    dyntrain_nn.add(tf.keras.layers.Dense(16, activation='tanh'))
    dyn_hyp = HyperParameters(lr=2e-2, k=100, frequency=8, scale=1, batch_size=20)
    dyn_training = DynamicsTraining(SWAG(), {"loss":tf.keras.losses.MeanSquaredError(), "likelihood": "Regression"},
        dyntrain_nn, dyn_hyp)

    env = gym.make("Acrobot")
    bnn = BayesianDynamics(
        env=env,
        horizon=25,
        dyn_training=dyn_training,
        policy=policy,
        rew_name="Acb 2 factors", # reward function in static/rewards.py
        learn_config=(25, 32, 0.95), # dynamic epochs factor, particle number, discount factor
    )
    dyn_training.compile_more(extra={"starting_model":dyn_training.model})

    bnn.learn(nb_epochs=5, record_file="static/results/learning.txt")

    # Run an interactive demo of the trained policy
    # Create the environment
    env = gym.make("Acrobot", render_mode="human")  # Use "human" render mode for visualization
    observation, info = env.reset(seed=42)
    done = False
    # Run the game loop
    print(">>Start real game")
    plt.title("Accumulative reward over time step")
    rewards, total_reward = [], 0
    states = []
    actions = []
    while not done:
        states.append(observation)
        # normalize observation -1 to 1 when taking policy; 
        # action take is the actual action taken without normalization
        action, action_take = policy.act(tf.reshape(policy.vec_normalize("obs", observation), (1,-1)))
        actions[].append(action_take[0])
        observation, reward, terminated, truncated, info = env.step(action_take[0].numpy())
        total_reward += reward  # Accumulate the reward
        rewards.append(total_reward)
        if terminated or truncated:
            done = True
        # You can add a delay here if the visualization is too fast
        time.sleep(0.02)

    print("--Game finished--")
    ts = range(len(rewards))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time step')
    ax1.set_ylabel('states (4 lines: b,r,g,y) actions (black dots)')
    for (c, s) in [('b', 0), ('r', 2), ('g', 4), ('y', 5)]:
        ax1.plot(ts, [state[s] for state in states], color=c)
    ax1.scatter(ts, actions, color='k')
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Total rewards (cyan line)')
    ax1.plot(ts, rewards, color='c')
    plt.show()
    env.close()  # Close the environment when done

# def test_srlz():
#     f = open("static/sessions/rl/continue/loss.pkl", "wb")
#     pickle.dump(tf.keras.losses.MeanSquaredError(), f)
#     f.close() 

    
