from src.dynamics.control import gym, Policy, Control, PolicyOptimizer
from src.datasets.Dataset import Dataset
from src.optimizers.Optimizer import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def complete_model(template:tf.keras.Sequential, ipd, opd, out_activation):
    # if type(ipd) == tuple:
    #     ipd = ipd[0]
    # print(template, ipd, opd, out_activation)
    
    
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=ipd))
    for layer in template.layers:
        network.add(layer)
    network.add(tf.keras.layers.Dense(opd, out_activation))   
    
    # print(network.layers)
    return network

class NNPolicy(Policy):
    # using tensorflow neural network for optimizing policy params "phi"
    def __init__(self, network, out_activation, hyperparams: dict):
        self.network = network # template network consisting of inner layers
        self.out_activation = out_activation
        self.hyperparams = hyperparams

    def setup(self, ipd, opd):
        # print("setting up", ipd, opd)
        self.network = complete_model(self.network, ipd, opd, self.out_activation)
        # print(self.network.layers)
        
        
    def optimize_step(self, loss, check_converge=False):
        with tf.GradientTape(persistent=True) as tape:
            grad = tape.gradient(loss, self.network.trainable_variables)
            weights = self.network.get_weights()
            new_weights = []
            for i in range(len(grad)):
                wg = tf.math.multiply(grad[i], self.hyperparams["lr"])
                m = tf.math.subtract(weights[i], wg)
                new_weights.append(m)
            self.network.set_weights(new_weights)

        # To be implemented: check convergence
        if check_converge:
            converge = False
            return converge

    def act(self, state):
        print(state, type(state), state.shape)
        # convert state to 2D tensor
        # state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = np.array(state).reshape(1, -1)  # Reshape the state

        action = self.network.predict([state])[0]
        # action needs to be a 1D tensor
        action = int(action[0])
        return action


class DynamicsTraining:
    # learn dynamic model f for state action transitions
    def __init__(self, optimizer:Optimizer, data_specs:list, 
                 template, out_activation, hyperparams, **kwargs):
        self.optimizer, self.template, self.out_activation = optimizer, template, out_activation
        self.hyperparams, self.rems = hyperparams, kwargs
        self.data_specs = data_specs
        self.start = False   

    def create_model(self, ipd, opd):
        model = complete_model(self.template, ipd, opd, self.out_activation) 
        self.model_config = model.to_json()    

    def train(self, features, targets, nb_epochs):
        data = tf.data.Dataset.from_tensor_slices((features, targets))
        dataset = Dataset(data, *self.data_specs)
        train_dataset = Dataset(
            dataset.train_data, *self.data_specs)
        if not self.start:
            self.optimizer.compile(self.hyperparams, self.model_config, 
                                   train_dataset, **self.rems)
            self.start = True
        else:
            self.optimizer._dataset = train_dataset
        self.optimizer.train(nb_epochs)


class BayesianDynamics(Control):
    def __init__(
        self, env: gym.Env, horizon:int, dyn_training:DynamicsTraining,
        policy: NNPolicy, state_reward, learn_config:tuple
    ):
        self.state_d = env.observation_space.shape
        self.action_d = env.action_space.shape
        self.action_fd = 1
        for d in env.action_space.shape:
            self.action_fd *= d
        self.state_fd = 1
        for d in self.state_d:
            self.state_fd *= d
        dyn_training.create_model(
            self.state_fd+self.action_fd, self.state_fd)
        policy.setup(self.state_d, self.action_fd)
        super().__init__(env, horizon, policy)
        # Bayesian Model optimizer learning state-action-transition "f"
        self.dyn_training = dyn_training
        self.state_reward = state_reward
        self.dyntrain_ep, self.kp, self.gamma = learn_config
        # self.policy_optimizer = policy_optimizer

    def sample_initial(self):
        # default sampling method
        print("initial sample from bayesian dynamics")
        return self.env.observation_space.sample()
    
    def execute(self):
        all_states, all_actions = super().execute()
        features = []
        targets = []
        for s in range(len(all_states)-1):
            feature = tf.concat([tf.reshape(all_states[s], [1, self.state_fd]),
                         tf.reshape(all_actions[s], [1, self.action_fd])], axis=1)
            features.append(feature)
            
            target = tf.reshape(all_states[s+1], [1, self.state_fd])
            targets.append(target)
            # features.append(tf.concat(tf.reshape(all_states[s], [self.state_fd]),
            #                 tf.reshape(all_actions[s], [self.action_fd])))
            # targets.append(tf.reshape(all_states[s+1], [self.state_fd]))
        return features, targets

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
        for t in self.horizon:
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
        tot_rew = 0
        discount = 1
        for states in trajectory:
            k_rew = 0
            for s in states:
                # if calculating cost, use negative state reward
                k_rew += self.state_reward(s)
            k_rew /= self.kp
            tot_rew += discount * k_rew
            discount *= self.gamma
        return tot_rew

    def learn(self, nb_epochs):
        def step(check_converge=False):
            # train dynamic model using transition dataset
            xs, ys = self.execute()
            self.dyn_training.train(xs, ys, self.dyntrain_ep)
            # predict trajectory
            traj = self.predict_trajectory(self.kp)
            # evaluate policy and trajectory
            tot_rew = self.reward(traj)
            # # optimize policy
            return self.policy.optimize_step(loss=-tot_rew, check_converge=check_converge)
            
            # optimize policy using the policy optimizer
            loss = -tot_rew  # Define the loss as the negative of the total reward
            policy_loss = self.policy_optimizer.optimize(self.policy, loss, check_converge)
            return policy_loss

        if nb_epochs:
            # learning for a given number of epochs
            for ep in range(nb_epochs):
                step()
        else:
            while not step(check_converge=True):
                # continue learning if policy not converge
                continue
