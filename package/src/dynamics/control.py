import tensorflow as tf
import gymnasium as gym
from abc import ABC, abstractmethod

class Policy(ABC):  # Policy optimizer
    def __init__(self):
        pass

    @abstractmethod
    def setup(self, **kwargs):
        pass

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
        self.horizon = horizon
        self.policy = policy
        self.policy.setup(env)

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
        for t in self.horizon:
            action = self.policy.act(state)
            state, reward, terminated, truncated, info = self.env.step(action)
            all_atates.append(state)
            all_actions.append(action)
        return all_atates, all_actions
    
    @abstractmethod
    def learn(self):
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




        
