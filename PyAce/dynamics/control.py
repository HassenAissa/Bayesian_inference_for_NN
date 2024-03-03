import tensorflow as tf
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

def space_flat(orig_shape):
    if orig_shape == ():
        return (1,)
    shape = 1
    for s in orig_shape:
        shape *= s
    return (shape,)

class Policy(ABC):  # Policy optimizer
    def __init__(self):
        self.dtype, self.range = None, None
    def setup(self, env: gym.Env):
        aspace = env.action_space
        self.action_d = aspace.shape
        self.action_fd = space_flat(aspace.shape)
        if isinstance(aspace, gym.spaces.Discrete):
            self.action_fd = (int(aspace.n),)
            self.oact = "softmax"
            self.range = (tf.convert_to_tensor(aspace.start), tf.convert_to_tensor(aspace.start+aspace.n-1))
            self.dtype = tf.int32
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
        pass

    @abstractmethod
    def act(self, state):
        pass
    
class PolicyOptimizer(ABC):
    @abstractmethod
    def compile(self, model, hyperparams):
        pass

    @abstractmethod
    def train(self, dataset, epochs):
        pass

    @abstractmethod
    def optimize(self, loss, check_convergence=True):
        pass

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
        print("First 3 states", all_states[:3], "\nLast 3 states", all_states[-3:], "\nactions", takes)
        return all_states, all_actions
    
    @abstractmethod
    def learn(self, nb_epochs, record):
        pass
    

class NNPolicyOptimizer(PolicyOptimizer):
    """Neural network policy optimizer for reinforcement learning
        model: tf.keras.Model - neural network model
        hyperparams: dict - hyperparameters for the optimizer
        convergence_threshold: float - threshold for convergence
        previous_loss: float - previous loss value, keeps track of previous loss for convergence check
    """

    def __init__(self, model: tf.keras.Model, hyperparams: dict):
        self.model = model
        self.hyperparams = hyperparams
        self.optimizer = tf.keras.optimizers.get(hyperparams.get("optimizer", "adam"))
        
        self.convergence_threshold = hyperparams.get("convergence_threshold", 1e-4)
        self.previous_loss = None

    def compile(self, model, hyperparams):
        # Compile the neural network model with the given hyperparameters
        self.model = model
        self.model.compile(optimizer=self.optimizer, loss=hyperparams["loss"])

    def train(self, dataset, epochs):
        # Train the model on the given dataset for a specified number of epochs
        self.model.fit(dataset, epochs=epochs)
        
    def check_convergence(self, loss):
        # Check for convergence
        if self.previous_loss is not None:
            # Check if absolute difference between previous and current loss is less than the threshold
            if abs(loss - self.previous_loss) < self.convergence_threshold:
                return True
        self.previous_loss = loss
        return False

    def optimize(self, loss, check_convergence=True):
        # Perform an optimization step to update the policy parameters
        with tf.GradientTape() as tape:
            # Calculate gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
        # Apply gradients to the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Check for convergence
        if check_convergence:
            return self.check_convergence(loss)