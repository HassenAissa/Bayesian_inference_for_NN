import numpy as np
import gymnasium as gym
import tensorflow as tf
import tensorflow_probability as tfp
import copy, json, pickle
from PyAce.datasets import Dataset
from PyAce.optimizers import Optimizer


# For debugging only
class DeepPilco:
    def __init__(
            self, env: gym.Env,
            policy, policy_optimizer, bayesian_model_json,
            bayesian_optimizer: Optimizer, bayesian_hyperparameters,
            n_optimizer_iterations: int, extra_parameters_for_optimizer: dict,
            reward, max_iterations: int = 7,
            T: int = 5, k_particles: int = 30, gamma: float = 1, horizon: int = 500
    ):
        self._policy = policy
        self._env = env
        self._T = T
        self._max_iterations = max_iterations
        self._k_particles = k_particles
        self._bayesian_optimizer = bayesian_optimizer
        self._bayesian_hyperparameters = bayesian_hyperparameters
        self._extra_parameters_for_optimizer = extra_parameters_for_optimizer
        self._bayesian_model_json = bayesian_model_json
        self._policy_optimizer = policy_optimizer
        self._n_optimizer_iterations = n_optimizer_iterations
        self._reward = reward
        self._gamma = gamma
        self._horizon = horizon
        self._total_targets = []
        self._total_features = []

    def _generate_k_particles(self, dtbn=None):
        # create k random bnn weights and k random inputs
        samples = []
        if dtbn == None:
            for i in range(self._k_particles):
                samples.append(self._env.reset(options = {"low": -0.18, "high" : 0.18})[0])
        else:
            for i in range(self._k_particles):
                samples.append(dtbn.sample())
        return tf.convert_to_tensor(samples)

    def _generate_k_models(self, bnn):
        models = []
        for i in range(self._k_particles):
            models.append(bnn.sample_model())
        return models

    def _execute(self, random=False):
        sample, info = self._env.reset(options = {"low": -0.18, "high" : 0.18})
        state = tf.convert_to_tensor(sample)
        all_actions = []
        all_states = [state]
        for h in range(self._horizon):
            state = tf.reshape(state, (1, -1))
            if random:
                action_taken = self._env.action_space.sample()
                action = tf.one_hot(action_taken, 2)
            else:
                action = self._policy(state)
                action_taken = tf.argmax(action, axis=1)
                action_taken = action_taken[0].numpy()
            print(action)
            state, reward, terminated, truncated, info = self._env.step(action_taken)
            if terminated or truncated:
                sample, info = self._env.reset(options = {"low": -0.18, "high" : 0.18})
            all_states.append(tf.convert_to_tensor(state))
            all_actions.append(action)

        for s in range(len(all_states) - 1):
            state = tf.reshape(all_states[s], (-1,))
            action = tf.reshape(all_actions[s], (-1,))
            action = tf.cast(action, dtype=state.dtype)
            feature = tf.concat([state, action], axis=0)
            target = all_states[s + 1]
            self._total_features.append(feature)
            self._total_targets.append(target)

        features = tf.convert_to_tensor(self._total_features)
        targets = tf.convert_to_tensor(self._total_targets)

        dataset = tf.data.Dataset.from_tensor_slices((features, targets))
        dataset = Dataset(dataset, loss=tf.keras.losses.MeanSquaredError,
                          likelihoodModel="Regression", target_dim=self._env.observation_space.shape,
                          train_proportion=1, feature_normalisation=True)
        return dataset

    def _expected_cost(self, states):
        cost = 0
        for s in states:
            s = tf.reshape(s, (-1,))
            # print(s, self._reward(s))
            cost += self._reward(s)
        cost = tf.math.divide(cost, len(states))
        return cost

    def step(self, ep, check_converge=False):

        print(">>Learning epoch", ep)
        # train dynamic model using transition dataset

        new_dataset = self._execute(ep < 0)

        optimizer = self._bayesian_optimizer()
        optimizer.compile(self._bayesian_hyperparameters, self._bayesian_model_json,
                          new_dataset, **self._extra_parameters_for_optimizer)
        optimizer.train(min(new_dataset.train_data.cardinality() * 100, self._n_optimizer_iterations))
        bayesian_model = optimizer.result()

        particles = self._generate_k_particles()
        models = self._generate_k_models(bayesian_model)

        gamma = self._gamma
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self._policy.trainable_variables)

            total_cost = self._expected_cost(particles)
            for t in range(self._T):
                predictions = []
                actions = self._policy(particles, training=True)
                for k in range(self._k_particles):
                    state = particles[k]
                    state = tf.reshape(state, (1, -1))
                    action = tf.reshape(tf.cast(actions[k], dtype=state.dtype), (1, -1))
                    state_and_action = tf.concat([state, action], axis=1)
                    pred = models[k](state_and_action)
                    # print(state_and_action)
                    # print(pred)
                    pred = tf.reshape(pred, (-1,))
                    predictions.append(pred)

                predictions = tf.convert_to_tensor(predictions)
                ymean = tf.math.reduce_mean(predictions, axis=0)
                ystd = tf.math.reduce_std(predictions, axis=0)
                dtbn = tfp.distributions.Normal(ymean, ystd)
                particles = self._generate_k_particles(dtbn)
                total_cost += gamma * self._expected_cost(particles)
                gamma *= gamma
        print(total_cost)
        grad = tape.gradient(total_cost, self._policy.trainable_variables)
        if grad is not None:
            # print(self._policy.trainable_variables)

            self._policy_optimizer.apply_gradients(zip(grad, self._policy.trainable_variables))
        return True

    def learn(self, nb_epochs):
        if nb_epochs:
            # learning for a given number of epochs
            for ep in range(1, nb_epochs + 1):
                self.step(ep)
        else:
            ep = 1
            while not self.step(ep, check_converge=True):
                # continue learning if policy not converge
                ep += 1
                continue
        print("--Learning completed--")

    def store(self, pref, tot_epochs):
        f = open(pref + "loss.pkl", "wb")
        pickle.dump(self.dyn_training.data_specs["loss"], f)
        f.close()
        info = dict()
        info["learn_config"] = (self.dyntrain_ep, self.kp, self.gamma)
        info["rew_name"] = self.rew_name
        info["horizon"] = self.horizon
        info["likelihood"] = self.dyn_training.data_specs["likelihood"]
        info["tot_epochs"] = tot_epochs
        f = open(pref + "agent.json", "w")
        json.dump(info, f)
        f.close()