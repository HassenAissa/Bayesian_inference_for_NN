from control import gym, Policy, Control
import tensorflow as tf
import numpy as np

class NNPolicy(Policy):
    # using tensorflow neural network for optimizing policy params "phi"
    def __init__(self, network: tf.keras.Model):
        self.network = network

    def act(self, state):
        return self.network(state)

class BayesianDynamics(Control):
    def __init__(self, env:gym.Env, n_episodes, policy, bnn):
        super.__init__(env, n_episodes, policy)
        self.bnn = bnn # Bayesian Model for learning state-action-transition "f"

    def k_particles(self, k):
        # create k random bnn weights and k random inputs
        self.kp = k
        self.models = []
        self.inputs = []
        for i in range(k):
            self.bnn._sample_weights()
            self.models.append(self.bnn._model.copy())
            self.inputs.append(self.sample_initial())
    
    def predict_trajectory(self, k):
        # Step 6 of pilco algorithm using k particles
        self.k_particles(k)
        xs = self.inputs
        ktraj = [xs]
        for t in self.n_episodes:
            ys = []
            for i in range(k):
                ys.append(self.models[i](xs[i]))
            ymean = np.mean(ys)
            yvar = np.var(ys)
            # To be implemented: sample input states xs for the next episode using Normal-ymean, yvar
            # xs = ...
            ktraj.append(xs)

    def reward(self, gamma, trajectory):
        # if calculating cost, use negative reward
        tot_rew = 0
        discount = 1
        for states in trajectory:
            k_rew = 0
            for s in states:
                k_rew = -self.state_reward(s)
            k_rew /= self.kp
            tot_rew += discount * k_rew
            discount *= gamma
        return tot_rew



    



