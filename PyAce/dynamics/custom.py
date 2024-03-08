import tensorflow as tf
from matplotlib import pyplot as plt

'''Reward functions'''

def h2(state):
    return state[0]*state[2] - state[1]*state[3]

def ht_speed(state, t):
    height = 4-state[0] - state[0]*2 - h2(state)
    speed = pow(state[4], 2)
    return height+speed

def upright(state, t):
    angle = - state[2] 
    rot = -state[3]*state[2]
    time = t*(-pow(angle,2) + pow(0.2095, 2))
    return angle+rot+time

'''Plotting functions'''
def plot_rewards(rewards, states=None, actions=None):
    plt.clf()
    pref = "static/results/"
    ts = range(len(rewards))
    plt.title("Rewards over time")
    plt.plot(ts, rewards)
    plt.savefig(pref+"reward.png")

def plot_acb(rewards, states, actions):
    pref = "static/results/"
    ts = range(len(rewards))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time step')
    ax1.set_ylabel('angles and actions (black dots)')
    for (c, s) in [('b', 0), ('r', 2)]:
        ax1.plot(ts, [state[s] for state in states], color=c)
    ax1.scatter(ts, actions, color='k')
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('speeds')
    for (c, s) in [('g', 4), ('y', 5)]:
        ax1.plot(ts, [state[s] for state in states], color=c)
    plt.savefig(pref+"record.png")
    plot_rewards(rewards)

def plot_cart(rewards, states, actions):
    pref = "static/results/"
    ts = range(len(rewards))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time step')
    ax1.set_ylabel('angle (blue line) and action (black dots)')
    ax1.plot(ts, [state[0] for state in states], color="b")
    ax1.scatter(ts, actions, color='k')
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('angular speed (red line)')
    ax1.plot(ts, [state[1] for state in states], color="r")
    plt.savefig(pref+"record.png")
    plot_rewards(rewards)

all_rewards = {"Acb 2 factors": ht_speed, "Cart": upright}
all_plots = {"Reward only": plot_rewards, "Acrobot plot": plot_acb, "CartPole plot": plot_cart}
