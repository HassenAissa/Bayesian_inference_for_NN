from .control import gym, Policy, Control,np
from PyAce.datasets import Dataset
from PyAce.optimizers import Optimizer
from static.custom import all_rewards
import tensorflow as tf
import tensorflow_probability as tfp
import copy, json, pickle
from tensorflow.keras import backend as bk

def complete_model(template:tf.keras.Sequential, ipd, opd, out_activation):
    network = tf.keras.Sequential()
    network.add(tf.keras.Input(shape=ipd))
    for layer in template.layers:
        network.add(layer)
    network.add(tf.keras.layers.Dense(opd[0], out_activation))   
    print("Network input output", ipd, opd)
    return network

class RBF(tf.keras.layers.Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBF, self).__init__(**kwargs)
        self.units = units
        self.gamma = bk.cast_to_floatx(gamma)
    
    def build(self, input_shape):
        self.mean = self.add_weight(name='mean',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBF, self).build(input_shape)

    def call(self, inputs):
        diff = bk.expand_dims(inputs) - self.mean
        norm = bk.sum(bk.pow(diff, 2), axis=1)
        return bk.exp(-1 * self.gamma * norm)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class NNPolicy(Policy):
    def __init__(self, network, hyperparams):
        super().__init__()
        # using tensorflow neural network for optimizing policy params "phi"
        self.network = network # template network consisting of inner layers
        self.hyperparams = hyperparams
        self.model_ready = False

    def setup(self, env: gym.Env, ipd):
        learning_rate = 1e-3
        if "lr" in self.hyperparams._params:
            learning_rate = self.hyperparams._params["lr"]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if not self.model_ready:
            print("Setup genral policy")
            Policy.setup(self, env)
            print("Setup NN policy")
            self.network = complete_model(self.network, ipd, self.action_fd, self.oact)
            self.model_ready = True
        
    def optimize_step(self, grad, check_converge=False):
        self.optimizer.apply_gradients(zip(grad, self.network.trainable_variables))

        # To be implemented: check convergence
        if check_converge:
            converge = False
            return converge
        
    def act(self, states, take=True):
        actions = self.network(states)
        action_takes = []
        if take:
            # Discrete action takes the maximum probability of all cases
            if self.oact == "softmax":
                for action in actions:
                    i = self.range[0]
                    max_a, max_p = i, action[0]
                    for p in action[1:]:
                        i += 1
                        if p > max_p:
                            max_a = i
                            max_p  = p
                    action_takes.append(tf.cast(max_a, self.dtype))
            else:
                for action in actions:
                    low = tf.math.maximum(action, self.range[0])
                    res = tf.math.minimum(low, self.range[1])
                    action_takes.append(tf.reshape(tf.cast(res, self.dtype), self.action_d))
        return actions, action_takes
                
class DynamicsTraining:
    # learn dynamic model f for state action transitions
    def __init__(self, optimizer:Optimizer, data_specs:dict, 
                 template=None, hyperparams=None):
        self.optimizer, self.template, = optimizer, template
        self.hyperparams = hyperparams
        self.data_specs = data_specs
        self.features, self.targets = [], []
        self.start = False   
        self.model_ready = (template is None)

    def create_model(self, ipd, opd):
        if self.model_ready:
            return
        model = complete_model(self.template, ipd, opd, out_activation="linear") 
        self.model = model   

    def compile_more(self, extra):
        self.rems = extra

    def train(self, features, targets, opd, ep_fac):
        if len(self.features)/len(features) > 50:
            self.targets = self.targets[len(self.features):]
            self.features = self.features[len(self.features):]
        self.features += features
        self.targets += targets
        print("Dyn data size:", len(self.features))
        data = tf.data.Dataset.from_tensor_slices((self.features, self.targets))
        train_dataset = Dataset(data, self.data_specs["loss"], self.data_specs["likelihood"], opd[0],
                                train_proportion=1.0, test_proportion=0.0, valid_proportion=0.0)
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
        # nb_epochs = int(ep_fac * np.sqrt(len(self.features)))
        # print("Dyn training epochs", nb_epochs)
        self.optimizer.train(ep_fac)

class BayesianDynamics(Control):
    def __init__(
        self, env: gym.Env, horizon:int, dyn_training:DynamicsTraining,
        policy: NNPolicy, rew_name, learn_config:tuple
    ):
        super().__init__(env, horizon, policy)
        self.policy.setup(self.env, self.state_d)
        ipd = (self.state_fd[0] + policy.action_fd[0],)
        opd = (self.state_fd[0],)
        dyn_training.create_model(ipd, opd)
        self.dyn_training = dyn_training
        self.rew_name = rew_name    
        self.state_reward = all_rewards[rew_name]
        if learn_config:
            self.dyntrain_ep, self.kp, self.gamma = learn_config
        # self.policy_optimizer = policy_optimizer
    
    def sample_initial(self):
        # default sampling method, return initial normalized states
        options = None  # {"low":-0.5, "high":0.5}
        sample, info = self.env.reset(options=options)  #{"low":-0.5, "high":0.5})
        return sample

    def dyn_feature(self, state0, action0):
        s0 = tf.reshape(state0, self.state_fd)
        feature = tf.concat([s0, action0], axis=0)
        return feature
    
    def dyn_target(self, state1):
        target = tf.reshape(state1, self.state_fd)
        return target
        
    def execute(self, use_policy=True):
        all_states, all_actions = super().execute(use_policy=use_policy)
        features = []
        targets = []
        for s in range(len(all_states)-1):
            feature = self.dyn_feature(all_states[s], all_actions[s])
            target = self.dyn_target(all_states[s+1])
            features.append(feature)
            targets.append(target)
        return features, targets

    def k_particles(self):
        # create k random bnn weights and k random inputs
        self.models = []
        samples = []
        bnn = self.dyn_training.optimizer.result()
        for i in range(self.kp):
            self.models.append(bnn.sample_model())
            samples.append(self.sample_initial())
        samples = tf.convert_to_tensor(samples) 
        return samples
    
    def forward(self, samples):  
        ys = []      
        actions, action_takes = self.policy.act(samples, take=False)
        for i in range(self.kp):
            feature = self.dyn_feature(samples[i], actions[i])
            y = self.models[i](tf.reshape(feature, shape=(1,-1))) # normalized new state
            ys.append(y[0])
        ys = tf.convert_to_tensor(ys)
        ymean = tf.math.reduce_mean(ys, axis=0)
        ystd = tf.math.reduce_std(ys, axis=0)
        dtbn = tfp.distributions.Normal(ymean, ystd)
        new_states = []
        for i in range(self.kp):
            x = dtbn.sample()
            new_states.append(x)
        return actions, tf.convert_to_tensor(new_states)  
        
    def t_reward(self, states, t):
        k_rew = 0
        for i in range(self.kp):
            # if calculating cost, use negative state reward
            k_rew += self.state_reward(states[i], t)
        exp_rew = k_rew / self.kp
        return exp_rew

    def learn(self, nb_epochs, record_file, random_ep):
        freq = max(int(self.horizon / 25), 1)
        if not random_ep:
            random_ep = 5
        else:
            random_ep = int(random_ep)
        def step(ep, check_converge=False):
            print(">>Learning epoch", ep)
            # train dynamic model using transition dataset
            use_policy = False
            if ep > random_ep:
                use_policy = True
            xs, ys = self.execute(use_policy=use_policy)
            self.dyn_training.train(xs, ys, self.state_fd, self.dyntrain_ep)
            if not use_policy:
                return
            
            # k sample inputs and k dynamic bnn
            states = self.k_particles()
            # predict trajectory and calculate gradient
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.policy.network.trainable_variables)
                tot_cost = -self.t_reward(states, 0)
                prev_tmark = 0
                discount = 1
                f = open(record_file, "a")
                f.write("Learning epoch "+str(ep)+"\n; Actions hor: ")
                for t in range(1,self.horizon+1):
                    actions, new_states = self.forward(states)
                    tmark = int(10*t/self.horizon)
                    if tmark > prev_tmark and tmark % 2 == 0:
                        print("Time step: "+str(t)+"/"+str(self.horizon))
                    prev_tmark = tmark
                    states = new_states
                    if t % freq == 0:
                        f.write(str([str(a.numpy())+"," for a in actions[:3]]))
                        discount *= self.gamma
                        tot_cost -= discount * self.t_reward(states, t)
                        
            grad = tape.gradient(tot_cost, self.policy.network.trainable_variables)
            f.write("\nTotal cost: "+str(tot_cost)+"\n")
            if None in grad:
                f.write("Invalid gradient!\n")
                f.close()
                return 
            f.write("Gradient sample: "+str(grad[-1])+", length "+str(len(grad))+"\n")
            f.close()
            return self.policy.optimize_step(grad, check_converge=check_converge)
        
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
