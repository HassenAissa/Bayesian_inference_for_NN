import gymnasium as gym

class Policy:
    def __init__(self, func):
        self.params = dict()
        self._func = func   # policy's functional form taking two arguments: params, state; returns action

    def get_action(self, state):
        return self._func(self.params, state)


class BayesianDynamics:
    def __init__(self, env:gym.Env, bnn, policy_form):
        self.observation, self.info = env.reset()
        self.bnn = bnn # from BayesianModel import BayesianModel
        self.policy = Policy(policy_form)

    def k_particles(self, k):
        self.kp = k
        self.models = []
        self.inputs = []
        for i in range(k):
            self.bnn._sample_weights()
            self.models.append(self.bnn._model.copy())
    
# example of simple policy y=kx+b
p = Policy()
p.params["k"] = 5
p.params["b"] = 2
func = lambda params, x: params["k"]*x + params["b"]
print(p.get_action(state=3, func=func)) # 5*3+2=17


