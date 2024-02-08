from src.dynamics.control import gym, Policy, Control,np
from src.datasets.Dataset import Dataset
from src.optimizers.Optimizer import Optimizer
import tensorflow as tf
import tensorflow_probability as tfp
import copy

def complete_model(template:tf.keras.Sequential, ipd, opd, out_activation):
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=ipd))
    for layer in template.layers:
        network.add(layer)
    network.add(tf.keras.layers.Dense(opd[0], out_activation))   
    return network

class NNPolicy(Policy):
    # using tensorflow neural network for optimizing policy params "phi"
    def __init__(self, network, out_activation, hyperparams):
        self.network = network # template network consisting of inner layers
        self.out_activation = out_activation
        self.hyperparams = hyperparams

    def setup(self, ipd, opd):
        print("Policy network input output dimensions:", ipd, opd)
        self.network = complete_model(self.network, ipd, opd, self.out_activation)
        
    def optimize_step(self, grad, check_converge=False):
        print("Policy optimization gradient", grad)
        if grad:
            weights = self.network.get_weights()
            new_weights = []
            for i in range(len(grad)):
                print("gradient", i, grad[i])
                wg = tf.math.multiply(grad[i], self.hyperparams.lr)
                m = tf.math.subtract(weights[i], wg)
                new_weights.append(m)
            self.network.set_weights(new_weights)

        # To be implemented: check convergence
        if check_converge:
            converge = False
            return converge

    def act(self, state):
        # convert state to 2D tensor
        # state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = self.network.predict([state])[0]
        action = np.int32(action)
        # action needs to be a 1D tensor
        return action


class DynamicsTraining:
    # learn dynamic model f for state action transitions
    def __init__(self, optimizer:Optimizer, data_specs:list, 
                 template, out_activation, hyperparams):
        self.optimizer, self.template, self.out_activation = optimizer, template, out_activation
        self.hyperparams = hyperparams
        self.data_specs = data_specs
        self.start = False   

    def create_model(self, sfd, afd):
        ipd = (sfd[0]+afd[0],)
        model = complete_model(self.template, ipd, sfd, self.out_activation) 
        self.model = model   

    def compile_more(self, **kwargs):
        self.rems = kwargs

    def train(self, features, targets, nb_epochs):
        data = tf.data.Dataset.from_tensor_slices((features, targets))
        dataset = Dataset(data, *self.data_specs)
        train_dataset = Dataset(
            dataset.train_data, *self.data_specs)
        if not self.start:
            self.optimizer.compile(self.hyperparams, self.model.to_json(), 
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
        super().__init__(env, horizon, policy)
        dyn_training.create_model(
            self.state_fd, self.action_fd)
        self.dyn_training = dyn_training
        self.policy.setup(self.state_d, self.action_fd)        
        self.state_reward = state_reward
        self.dyntrain_ep, self.kp, self.gamma = learn_config
        # self.policy_optimizer = policy_optimizer

    def sample_initial(self):
        # default sampling method
        return self.env.observation_space.sample().tolist()
    
    def dyn_feature(self, state0, action0):
        s0 = tf.reshape(state0, self.state_fd)
        a0 = tf.reshape(action0, self.action_fd)
        feature = s0.numpy().tolist()
        for a in a0.numpy():
            feature.append(a)
        return feature
    
    def dyn_target(self, state1):
        target = tf.reshape(state1, self.state_fd)
        return target.numpy().tolist()
        
    def execute(self):
        all_states, all_actions = super().execute()
        features = []
        targets = []
        for s in range(len(all_states)-1):
            feature = self.dyn_feature(all_states[s], all_actions[s])
            target = self.dyn_target(all_states[s+1])
            features.append(feature)
            targets.append(target)
        return tf.constant(features),tf.constant(targets)

    def k_particles(self):
        # create k random bnn weights and k random inputs
        self.models = []
        samples = []
        bnn = self.dyn_training.optimizer.result()
        for i in range(self.kp):
            bnn._sample_weights()
            self.models.append(copy.deepcopy(bnn._model))
            samples.append(self.sample_initial())
        return samples

    def predict_trajectory(self):
        # Step 6 of pilco algorithm using k particles
        xs = self.k_particles()
        traj = [xs]
        for t in range(self.horizon):
            ys = []
            print("Time step: "+str(t)+"/"+str(self.horizon))
            for i in range(self.kp):
                state = xs[i]
                action = self.policy.act(state)
                feature = self.dyn_feature(state, action)
                ys.append(self.models[i].predict([feature])[0].tolist())
            ys = np.array(ys)
            ymean = np.mean(ys, axis=0)
            ystd = np.std(ys, axis=0)
            dtbn = tfp.distributions.Normal(ymean, ystd)
            xs = []
            for i in range(self.kp):
                x = dtbn.sample()
                xs.append(x.numpy().tolist())
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
        def calc_loss():
            # train dynamic model using transition dataset
            xs, ys = self.execute()
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.policy.network.trainable_variables)
            self.dyn_training.train(xs, ys, self.dyntrain_ep)
            # predict trajectory
            traj = self.predict_trajectory()
            tot_rew = self.reward(traj)
            loss = tf.constant(-tot_rew)
            return loss
        
        def step(check_converge=False):
            # with tf.GradientTape(persistent=True) as tape:
            #     tape.watch(self.policy.network.trainable_variables)
            #     loss = tf.constant(calc_loss())
            # grad = tape.gradient(loss, self.policy.network.trainable_variables)
            loss = tf.constant(calc_loss())
            print("Loss value:", loss)
            grad = None

            return self.policy.optimize_step(grad, check_converge=check_converge)
            

        if nb_epochs:
            # learning for a given number of epochs
            for ep in range(nb_epochs):
                print(">>Learning epoch", ep)
                step()
        else:
            while not step(check_converge=True):
                # continue learning if policy not converge
                continue
        print("--Learning completed--")
