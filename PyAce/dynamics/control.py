import tensorflow as tf
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod
from . import Policy


class Control(ABC):
    # Reinforcement learning basics using gymnasium
    def __init__(self, env:gym.Env, horizon:int, policy:Policy):
        self.env = env
        self.state_d = env.observation_space.shape
        self.state_fd = space_flat(self.state_d)
        self.horizon = horizon
        self.policy = policy
        # self.policy.setup(env)

    @abstractmethod
    def sample_initial(self):
        # sample from an initial state distribution
        pass

    @abstractmethod
    def t_reward(self, **kwargs):
        # if calculating cost, reward is negative
        pass

    def reward(self, **kwargs):
        discount = 1
        tot_rew = 0
        for t in self.horizon:
            rew = self.t_reward()
            tot_rew += discount * rew
            discount *= self.gamma
        return tot_rew

    def execute(self):
        # take actions according to policy for n episodes, rest env every time for initial states
        state = self.sample_initial()
        print("Main trial initial state", state)
        all_states = [tf.convert_to_tensor(state)]
        all_actions,takes = [],[]
        for t in range(self.horizon):
            actions, action_takes = self.policy.act(tf.reshape(state, (1,-1)))
            state, reward, terminated, truncated, info = self.env.step(action_takes[0].numpy())
            all_states.append(tf.convert_to_tensor(state))
            all_actions.append(actions[0])
            takes.append(action_takes[0].numpy())
        # print("First 3 states", all_states[:3], "\nLast 3 states", all_states[-3:], "\nactions", takes)
        return all_states, all_actions
    
    @abstractmethod
    def learn(self, nb_epochs, record):
        pass
    

