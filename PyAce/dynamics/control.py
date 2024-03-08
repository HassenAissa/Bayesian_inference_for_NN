import tensorflow as tf
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

def space_flat(orig_shape):
    '''
    Flatten the observation/action space to one dimension
    Args:
        orig_shape (tuple): original space dimansion
    '''
    if orig_shape == ():
        return (1,)
    shape = 1
    for s in orig_shape:
        shape *= s
    return (shape,)

class Policy(ABC):  # Policy optimizer
    '''
    Base class representing policy for any reinforcement learning algorithm
    '''
    def __init__(self):
        self.dtype, self.range = None, None
    def setup(self, env: gym.Env):
        '''
        Given gym environment, find the action space and acceptable range, and determine output structure
        If action is discrete, use softmax for every value; 
        otherwise, use relu or linear depending on whether negative number is taken
        '''
        aspace = env.action_space
        self.action_d = aspace.shape
        self.action_fd = space_flat(aspace.shape)
        if isinstance(aspace, gym.spaces.Discrete):
            self.action_fd = (int(aspace.n),)
            self.oact = "softmax"
            self.range = (tf.convert_to_tensor(aspace.start), tf.convert_to_tensor(aspace.start+aspace.n-1))
        elif isinstance(aspace, gym.spaces.Box):
            low = None
            if isinstance(aspace.low, np.ndarray):
                low = min(aspace.low)
            else:
                low = aspace.low
            if low >= 0:
                self.oact = "relu"
            else:
                self.oact = "linear"
            self.range = (tf.convert_to_tensor(aspace.low), tf.convert_to_tensor(aspace.high))
        self.dtype = aspace.dtype

    @abstractmethod
    def optimize_step(self, **kwargs):
        '''
        Take one step to optimize policy
        '''
        pass

    @abstractmethod
    def act(self, states):
        '''
        Determine the actions given states
        '''
        pass

class Control(ABC):
    '''
    Reinforcement learning basic class using Gym
    inputs:
        environment, time horizon, policy model
    '''
    def __init__(self, env:gym.Env, horizon:int, policy:Policy):
        self.env = env
        self.state_d = env.observation_space.shape
        self.state_fd = space_flat(self.state_d)
        self.horizon = horizon
        self.policy = policy
        # self.policy.setup(env)

    @abstractmethod
    def sample_initial(self):
        '''
        sample a set of initial states representing probability distribution
        '''
        pass

    @abstractmethod
    def t_reward(self, **kwargs):
        '''
        Expected reward of a time step
        '''
        pass

    def reward(self, **kwargs):
        '''
        Total reward over all time steps
        '''
        discount = 1
        tot_rew = 0
        for t in self.horizon:
            rew = self.t_reward()
            tot_rew += discount * rew
            discount *= self.gamma
        return tot_rew
    
    def random_action(self):
        '''
        Use random number generator to generate a list of random actions
        '''
        aspace = self.env.action_space
        import random
        if isinstance(aspace, gym.spaces.Discrete):
            probs = []
            for i in range(aspace.n):
                probs.append(random.random())
            probs.sort()
            action = [probs[0]]
            for i in range(1, aspace.n):
                action.append(probs[i]-probs[i-1])
            action = tf.convert_to_tensor(action)
            action_take = tf.argmax([action], axis=1)
            return action, action_take[0]
        elif isinstance(aspace, gym.spaces.Box):
            action = tf.convert_to_tensor(aspace.sample()) 
            return tf.cast(action, self.env.observation_space.dtype), action
    
    def execute(self, use_policy=True):
        '''
        Take actions according to policy for time horizon, resett env every time for initial states
        '''
        state = self.sample_initial()
        print("Main trial initial state", state)
        all_states = [tf.convert_to_tensor(state)]
        all_actions,takes = [],[]
        for t in range(self.horizon):
            action, action_take = None, None
            if use_policy:
                actions, action_takes = self.policy.act(tf.reshape(state, (1,-1)))
                action = actions[0]
                action_take = action_takes[0]
            else:
                action, action_take = self.random_action()
            state, reward, terminated, truncated, info = self.env.step(action_take.numpy())
            all_states.append(tf.convert_to_tensor(state))
            all_actions.append(action)
            takes.append(action_take.numpy())
        # print("First 3 states", all_states[:3], "\nLast 3 states", all_states[-3:], "\nactions", takes)
        print(takes)
        return all_states, all_actions
    
    @abstractmethod
    def learn(self, nb_epochs, record):
        pass
    
class PolicyOptimizer(ABC):
    pass
