import tensorflow as tf
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

def _space_flat(orig_shape):
    if orig_shape == ():
        return (1,)
    shape = 1
    for s in orig_shape:
        shape *= s
    return (shape,)

class Policy(ABC):  # Policy optimizer
    '''
    Base class representing policy for any reinforcement learning algorithm.
    User defines the specific form of policy model (e.g., neural networks) by inheritance.
    '''
    def __init__(self):
        self.dtype, self.range = None, None
    def setup(self, env: gym.Env):
        '''
        Complete the class after initialization: find action range and shape information.
        Args:
            env: Gym environment
            ipd: input dimension
        '''
        aspace = env.action_space
        self.action_d = aspace.shape
        self.action_fd = _space_flat(aspace.shape)
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
    def _optimize_step(self, **kwargs):
        pass

    @abstractmethod
    def act(self, states):
        '''
        Determine the actions given a set of states
        Args:
            states (list(tensors))
        '''
        pass

class Control(ABC):
    '''
    Reinforcement learning controller base class using Gym
    inputs:
        environment: gym environment after "make"
        horizon (int): time horizon
        policy: the policy model for learning
    '''
    def __init__(self, env:gym.Env, horizon:int, policy:Policy):
        self.env = env
        self.state_d = env.observation_space.shape
        self.state_fd = _space_flat(self.state_d)
        self.horizon = horizon
        self.policy = policy
        # self.policy.setup(env)

    @abstractmethod
    def _sample_initial(self):

        pass

    @abstractmethod
    def _t_reward(self, **kwargs):
        pass

    def reward(self, **kwargs):
        '''
        Total reward over all time steps
        '''
        discount = 1
        tot_rew = 0
        for t in self.horizon:
            rew = self._t_reward()
            tot_rew += discount * rew
            discount *= self.gamma
        return tot_rew
    
    def _random_action(self):
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
    
    def _execute(self, use_policy=True):
        state = self._sample_initial()
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
                action, action_take = self._random_action()
            state, reward, terminated, truncated, info = self.env.step(action_take.numpy())
            all_states.append(tf.convert_to_tensor(state))
            all_actions.append(action)
            takes.append(action_take.numpy())
        # print("First 3 states", all_states[:3], "\nLast 3 states", all_states[-3:], "\nactions", takes)
        print(takes)
        return all_states, all_actions
    
    @abstractmethod
    def learn(self, nb_epochs, **kwargs):
        '''
        The main procedure of reinforcement learning algorithm
        Args:
            nb_epochs (int): the number of iterations to run the algorithm
        '''
        pass
    
class PolicyOptimizer(ABC):
    pass
