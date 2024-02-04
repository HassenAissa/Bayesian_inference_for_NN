import gymnasium as gym
from abc import ABC, abstractmethod

class Policy(ABC):  # Policy optimizer
    def __init__(self):
        pass
    def setup(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    @abstractmethod
    def optimize_step(self, **kwargs):
        pass

    @abstractmethod
    def act(self, state):
        pass

class Control(ABC):
    # Reinforcement learning basics using gymnasium
    def __init__(self, env:gym.Env, n_episodes:int, policy:Policy, state_reward):
        self.env = env
        self.n_episodes = n_episodes
        state_dim = env.observation_space.shape
        action_dim = env.action_space.shape
        self.policy = policy
        self.policy.setup(state_dim,action_dim)
        self.state_reward = state_reward

    @abstractmethod
    def sample_initial(self):
        # sample from an initial state distribution
        pass

    @abstractmethod
    def reward(self, **kwargs):
        # if calculating cost, reward is negative
        pass

    def execute(self):
        # take actions according to policy for n episodes
        state = self.sample_initial() # initial state
        all_atates = [state]
        all_actions = []
        for t in self.n_episodes:
            action = self.policy.act(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            all_atates.append(state)
            all_actions.append(action)
        return all_atates, all_actions
    
    @abstractmethod
    def learn(self):
        pass




        
