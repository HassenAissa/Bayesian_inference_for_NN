import tensorflow as tf
import gymnasium as gym
import numpy as np
from abc import ABC, abstractmethod

class Policy(ABC):  # Policy optimizer
    def __init__(self):
        self.dtype, self.start, self.range = [],[],[]

    def setup(self, env: gym.Env):
        self.act_dim = env.action_space.shape
        for space in [env.observation_space, env.action_space]:
            # Use original shapes not flatten
            self.dtype.append(space.dtype)
            if isinstance(space, gym.spaces.Discrete):
                self.start.append(tf.convert_to_tensor(space.start, float))
                self.range.append(tf.convert_to_tensor(space.n-1, float))
            elif isinstance(space, gym.spaces.Box):
                low, high = None, None
                if isinstance(space.low, np.ndarray):
                    low = tf.convert_to_tensor(space.low, float)
                else:
                    low = tf.fill(space.shape, float(space.low))
                if isinstance(space.high, np.ndarray):
                    high = tf.convert_to_tensor(space.high, float)
                else:
                    high = tf.fill(space.shape, float(space.high))
                self.start.append(low)
                self.range.append(tf.subtract(high, low))

    def vec_normalize(self, mode, vec:tf.Tensor):
        m = 0   # mode == "obs"
        if mode == "act":
            m = 1
        diff = tf.math.subtract(tf.cast(vec, float), self.start[m])
        res = tf.divide(diff, self.range[m]) * 2 - 1    # from -1 to 1, match tanh
        return res

    def norm_restore(self, mode, norm:tf.Tensor):
        m = 0   # mode == "obs"
        if mode == "act":
            m = 1
        diff = tf.multiply((norm + 1) / 2, self.range[m])
        orig = tf.math.add(diff, self.start[m])
        dtype = self.dtype[m]
        if tf.cast(1.2, dtype) == 1:
            orig = tf.cast(tf.round(orig), dtype) 
        return orig

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
        self.action_d = env.action_space.shape
        self.action_fd = Control.space_flat(self.action_d)
        self.state_fd = Control.space_flat(self.state_d)
        print("Observation and action space/flatten",
            self.state_d, self.state_fd, self.action_d, self.action_fd)
        self.horizon = horizon
        self.policy = policy
        # self.policy.setup(env)
    def space_flat(orig_shape):
        if orig_shape == ():
            return (1,)
        shape = 1
        for s in orig_shape:
            shape *= s
        return (shape,)

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
        # take actions according to policy for n episodes
        state = self.sample_initial() # initial state
        all_atates = [state]
        all_actions = []
        for t in range(self.horizon):
            action, action_take = self.policy.act(state)
            # action = action.numpy()
            state, reward, terminated, truncated, info = self.env.step(action_take)
            state = self.policy.vec_normalize("obs", tf.convert_to_tensor(state))
            all_atates.append(state)
            all_actions.append(action)
        return all_atates, all_actions
    
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




        
