from control import gym, Policy, Control
from src.datasets.Dataset import Dataset
from src.optimizers.Optimizer import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class NNPolicy(Policy):
    # using tensorflow neural network for optimizing policy params "phi"
    def __init__(self, network: tf.keras.Model):
        self.network = network

    def act(self, state):
        return self.network(state)

class DynamicsTraining:
    # learn dynamic model f for state action transitions
    def __init__(self, optimizer:Optimizer, hyperparams, base_model, prior, loss, learn_type, normalise):
        self.optimizer = optimizer
        self.optimizer.compile(hyperparams, base_model, dataset=None, prior=prior)
        
        self.loss = loss    # loss function
        self.learn_type = learn_type    # classification/regression
        self.normalise = normalise
        
        self.start = False
        self.ipdim = 1
        self.tgdim = 1

    def add_transitions(self, all_states, all_actions):
        if not self.start:
            for s in all_states.shape:
                self.ipdim *= s
            for s in all_actions.shape:
                self.tgdim *= s
            self.start = True

        features = []
        targets = []
        for s in range(len(all_states)-1):
            features.append(tf.concat(tf.reshape(all_states[s], [self.ipdim]), 
                                       tf.reshape(all_states[s+1], [self.ipdim])))
            targets.append(tf.reshape(all_actions[s], self.tgdim))
        
        data = tf.data.Dataset.from_tensor_slices((features, targets))
        dataset = Dataset(data, self.loss, self.learn_type, self.normalise)
        self.train_dataset = Dataset(dataset.train_data, tf.keras.losses.MeanSquaredError(), "Regression")

    def train(self, nb_epochs):
        self.optimizer._dataset = self.train_dataset
        self.optimizer.train(nb_epochs)

class BayesianDynamics(Control):
    def __init__(self, env:gym.Env, n_episodes, policy, state_reward, dyntrain_config:list, learn_config:tuple):
        super.__init__(env, n_episodes, policy, state_reward)
        self.training = DynamicsTraining(*dyntrain_config)  # Bayesian Model optimizer learning state-action-transition "f"
        self.nb_epochs, self.kp, self.gamma = learn_config
    
    def sample_initial(self):
        # default sampling method
        print("init sample from bayesian dynamics")
        return self.env.observation_space.sample()
    
    def k_particles(self, k):
        # create k random bnn weights and k random inputs
        self.models = []
        self.inputs = []
        bnn = self.training.optimizer.result()
        for i in range(k):
            bnn._sample_weights()
            self.models.append(bnn._model.copy())
            self.inputs.append(self.sample_initial())
    
    def predict_trajectory(self, k):
        # Step 6 of pilco algorithm using k particles
        self.k_particles(k)
        xs = self.inputs
        traj = [xs]
        for t in self.n_episodes:
            ys = []
            for i in range(k):
                ys.append(self.models[i](xs[i]))
            ymean = np.mean(ys)
            ystd = np.std(ys)
            dtbn = tfp.distributions.Normal(ymean, ystd)
            xs = []            
            for i in range(k):
                x = dtbn.sample()
                xs.append(x)
            traj.append(xs)
        return traj

    def reward(self, trajectory):
        # if calculating cost, use negative state reward
        tot_rew = 0
        discount = 1
        for states in trajectory:
            k_rew = 0
            for s in states:
                k_rew = self.state_reward(s)
            k_rew /= self.kp
            tot_rew += discount * k_rew
            discount *= self.gamma
        return tot_rew
    
    def learn(self):
        while True:
            # train dynamic model using transition dataset
            all_states, all_actions = self.execute()
            self.training.add_transitions(all_states, all_actions)
            self.training.train(self.nb_epochs)
            # predict trajectory
            traj = self.predict_trajectory(self.kp)
            # evaluate policy and trajectory
            tot_rew = self.reward(traj)
            # optimize policy



            
            

