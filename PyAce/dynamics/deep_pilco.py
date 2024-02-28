from .control import gym, Policy, Control,np
from PyAce.datasets import Dataset
from PyAce.optimizers import Optimizer
from static.rewards import all_rewards
import tensorflow as tf
import tensorflow_probability as tfp
import copy, json, pickle

def complete_model(template:tf.keras.Sequential, ipd, opd, out_activation):
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=ipd))
    for layer in template.layers:
        network.add(layer)
    network.add(tf.keras.layers.Dense(opd[0], out_activation))   
    return network

class NNPolicy(Policy):
    def __init__(self, network, hyperparams):
        super().__init__()
        # using tensorflow neural network for optimizing policy params "phi"
        self.network = network # template network consisting of inner layers
        self.hyperparams = hyperparams
        self.model_ready = False

    def setup(self, env: gym.Env, ipd, opd):
        print("Setup NN policy")
        if self.model_ready:
            return
        self.network = complete_model(self.network, ipd, opd, "tanh")
        self.model_ready = True
        print("Setup genral policy")
        Policy.setup(self, env)
        
    def optimize_step(self, grad, check_converge=False):
        weights = self.network.get_weights()
        new_weights = []
        for i in range(len(grad)):
            wg = tf.math.multiply(grad[i], self.hyperparams.lr)
            m = tf.math.subtract(weights[i], wg)
            new_weights.append(m)
        self.network.set_weights(new_weights)

        # To be implemented: check convergence
        if check_converge:
            converge = False
            return converge
        
    def act(self, state, training=False):
        action = self.network(tf.reshape(state, shape=(1, -1)), training=training)
        action = tf.reshape(action, shape=(-1,))        # normalized, standard shape
        # action needs to be a 1D tensor
        action_take = None
        if not training:
            action_take = self.norm_restore("act", tf.reshape(action, self.act_dim))  # restore, original shape
        return action, action_take

class DynamicsTraining:
    # learn dynamic model f for state action transitions
    def __init__(self, optimizer:Optimizer, data_specs:dict, 
                 template=None, hyperparams=None):
        self.optimizer, self.template, = optimizer, template
        self.hyperparams = hyperparams
        self.data_specs = data_specs
        self.start = False   
        self.model_ready = (template is None)

    def create_model(self, sfd, afd):
        if self.model_ready:
            return
        ipd = (sfd[0]+afd[0],)
        model = complete_model(self.template, ipd, sfd, "tanh") 
        self.model = model   

    def compile_more(self, extra):
        self.rems = extra

    def train(self, features, targets, opd, nb_epochs):
        data = tf.data.Dataset.from_tensor_slices((features, targets))
        train_dataset = Dataset(data, self.data_specs["loss"], self.data_specs["likelihood"], opd[0])
        # train_dataset = Dataset(
        #     dataset.train_data, self.data_specs["loss"], self.data_specs["likelihood"], opd)
        if not self.start:
            try:
                self.optimizer.compile(self.hyperparams, self.model.to_json(), 
                                    train_dataset, **self.rems)
            except:
                self.optimizer._dataset = train_dataset
                self.optimizer._dataset_setup()
            self.start = True
        else:
            self.optimizer._dataset = train_dataset
        self.optimizer.train(nb_epochs)

class BayesianDynamics(Control):
    def __init__(
        self, env: gym.Env, horizon:int, dyn_training:DynamicsTraining,
        policy: NNPolicy, rew_name, learn_config:tuple
    ):
        super().__init__(env, horizon, policy)
        dyn_training.create_model(self.state_fd, self.action_fd)
        self.dyn_training = dyn_training
        self.policy.setup(self.env, self.state_d, self.action_fd)    
        self.rew_name = rew_name    
        self.state_reward = all_rewards[rew_name]
        if learn_config:
            self.dyntrain_ep, self.kp, self.gamma = learn_config
        # self.policy_optimizer = policy_optimizer
    
    def sample_initial(self):
        # default sampling method, return initial normalized states
        sample = self.env.observation_space.sample()
        res = tf.convert_to_tensor(sample) 
        return self.policy.vec_normalize("obs", res)
    
    def dyn_feature(self, state0, action0):
        s0 = tf.reshape(state0, self.state_fd)
        feature = tf.concat([s0, action0], axis=0)
        return feature
    
    def dyn_target(self, state1):
        target = tf.reshape(state1, self.state_fd)
        return target
        
    def execute(self):
        all_states, all_actions = super().execute()
        features = []
        targets = []
        for s in range(len(all_states)-1):
            feature = self.dyn_feature(all_states[s], all_actions[s])
            target = self.dyn_target(all_states[s+1])
            features.append(feature)
            targets.append(target)
        return tf.convert_to_tensor(features),tf.convert_to_tensor(targets)

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
    
    def forward(self, samples):  
        ys = []      
        actions = []
        for i in range(self.kp):
            state = samples[i]      # normalized state
            action, action_take = self.policy.act(state, training=True) # normalized act
            actions.append(action)
            feature = self.dyn_feature(state, action)
            y = self.models[i](tf.reshape(feature, shape=(1,-1))) # normalized new state
            ys.append(tf.reshape(y, shape=(-1,)))
        ys = tf.convert_to_tensor(ys)
        ymean = tf.math.reduce_mean(ys, axis=0)
        ystd = tf.math.reduce_std(ys, axis=0)
        dtbn = tfp.distributions.Normal(ymean, ystd)
        new_states = []
        for i in range(self.kp):
            x = dtbn.sample()
            new_states.append(x)
        return ys, actions, new_states 
        
    def t_reward(self, ys, actions, t):
        k_rew = 0
        for i in range(self.kp):
            # if calculating cost, use negative state reward
            k_rew += self.state_reward(ys[i], actions[i], t)
        exp_rew = k_rew / self.kp
        return exp_rew

    def learn(self, nb_epochs, record_file):
        def step(ep, check_converge=False):
            print(">>Learning epoch", ep)
            # train dynamic model using transition dataset
            xs, ys = self.execute()
            self.dyn_training.train(xs, ys, self.state_fd, self.dyntrain_ep)
            # k sample inputs and k dynamic bnn
            states = self.k_particles()
            # predict trajectory and calculate gradient
            discount = 1
            tot_grad = None
            tot_loss = 0
            prev_tmark = 0
            f = open(record_file, "a")
            f.write("Learning epoch "+str(ep)+"; actions hor avg: ")
            for t in range(self.horizon):
                with tf.GradientTape(persistent=True) as tape:
                    ys, actions, new_states = self.forward(states)
                    l = -self.t_reward(ys, actions, t)
                    loss = tf.fill([1], l)
                    f.write(str(tf.reduce_mean(actions).numpy())+",")
                grad = tape.gradient(loss, self.policy.network.trainable_variables)
                tmark = int(10*t/self.horizon)
                if tmark > prev_tmark and tmark % 2 == 0:
                    print("Time step: "+str(t)+"/"+str(self.horizon))
                prev_tmark = tmark
                if not tot_grad:
                    tot_grad = grad
                    tot_loss = l
                elif None not in grad: 
                    for g in range(len(grad)):
                        tot_grad[g] = tf.math.add(tot_grad[g], tf.math.multiply(grad[g], discount)) 
                    tot_loss += l * discount
                discount *= self.gamma
                states = new_states
            f.write("\nTotal loss: "+str(tot_loss)+"\n")
            if None in tot_grad:
                f.write("Invalid gradient!\n")
                f.close()
                return 
            f.write("Gradient sample: "+str(tot_grad[-1])+"\n")
            f.close()
            return self.policy.optimize_step(tot_grad, check_converge=check_converge)
        
        f = open(record_file, "w")
        f.write("")
        f.close()
        if nb_epochs:
            # learning for a given number of epochs
            for ep in range(1, nb_epochs+1):
                step(ep)
        else:
            ep = 1
            while not step(ep, check_converge=True):
                # continue learning if policy not converge
                ep += 1
                continue
        print("--Learning completed--")

    def store(self, pref, tot_epochs):
        f = open(pref+"loss.pkl", "wb")
        pickle.dump(self.dyn_training.data_specs["loss"], f)
        f.close()
        info = dict()
        info["learn_config"] = (self.dyntrain_ep, self.kp, self.gamma)
        info["rew_name"] = self.rew_name
        info["horizon"] = self.horizon
        info["likelihood"] = self.dyn_training.data_specs["likelihood"]
        info["tot_epochs"] = tot_epochs
        f = open(pref+"agent.json", "w")
        json.dump(info, f)
        f.close()
