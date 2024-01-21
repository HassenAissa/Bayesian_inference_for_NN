from optimizer import Optimizer
import sys
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src')
from hyperparameters import Hyperparams
import copy
import torch
import utils
import math

class SwagHyperparam(Hyperparams):
    def __init__(self, loss, k=20, frequency=5, scale = 1, lr = 1e-1):
        self.k = k
        self.frequency = frequency
        self.scale = scale
        self.lr = lr
        self.loss = loss

class SWAG(Optimizer):
    def __init__(self,base_model, dataloader, hyperparameters):
        super(SWAG, self).__init__(hyperparameters)
        self.params = list()
        self.n_models = 1
        self.base = copy.deepcopy(base_model)
        self.base.apply(lambda module: self.swag_parameters(module=module, params=self.params))
        self.dataloader = dataloader
        self.hyperparameters = hyperparameters
        self.base_model = base_model
        self.base_model_optimizer = torch.optim.SGD(self.base_model.parameters(), lr= self.hyperparameters.lr)


    def swag_parameters(self, module, params):
        for name in list(module._parameters.keys()):
            if module._parameters[name] is None:
                continue
            data = module._parameters[name].data
            module._parameters.pop(name)
            module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
            module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())
            module.register_buffer("%s_deviation_matrix" % name, data.new_empty((0, data.numel())).zero_())

            params.append((module, name))

    def update_hyperparam():
        pass

    def hyperparam_init():
        pass

    def distribution(self):
        mean_list = []
        sq_mean_list = []
        deviation_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)
            mean_list.append(mean)
            sq_mean_list.append(sq_mean)
            deviation = module.__getattr__("%s_deviation_matrix" % name)
            deviation_list.append(deviation)

        mean = utils.flatten(mean_list)
        sq_mean = utils.flatten(sq_mean_list)
        deviation = torch.cat(deviation_list, dim=1)
        # deviation_matrix = deviation.T.matmul(deviation)

        distribution = torch.distributions.MultivariateNormal
        #     (mean, 0.5 * torch.diag(sq_mean-mean ** 2.0) +
        #     self.hyperparameters.scale * 0.5 * deviation_matrix / (self.hyperparameters.k-1)
        # )
        return mean, sq_mean-mean ** 2.0, deviation
        # sample = distribution.sample()
        # sample = sample.unsqueeze(0)
        # samples_list = utils.unflatten_like(sample, mean_list)
        # for (module, name), sample in zip(self.params, samples_list):
        #     module.__setattr__(name, sample)
        

    def predict(self, mean, var, deviation, input):
        

        mean_list = []

        for (module, name) in self.params:
            mean_i = module.__getattr__("%s_mean" % name)
            mean_list.append(mean_i)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)
        cov_sample = deviation.t().matmul(
            deviation.new_empty(
                (deviation.size(0),), requires_grad=False
            ).normal_()
        )

        sample = mean + math.sqrt(self.hyperparameters.scale) * (math.sqrt(0.5)*(cov_sample/((self.n_models-1)**0.5) + var_sample))
        sample = sample.unsqueeze(0)

        # print(sample.shape)
        samples_list = utils.unflatten_like(sample, mean_list)
        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample)
        return self.base(input)
            

    def step(self):
        self.base_model.train()
        self.base_model.zero_grad()
        input, label = next(iter(self.dataloader))
        loss = self.hyperparameters.loss(self.base_model(input), label)
        loss.backward()
        self.base_model_optimizer.step()
        for (module, name), base_param in zip(self.params, self.base_model.parameters()):
            if self.n_models % self.hyperparameters.frequency == 0:
                mean = module.__getattr__("%s_mean" % name)
                sq_mean = module.__getattr__("%s_sq_mean" % name)

                mean = mean * self.n_models / (
                    self.n_models + 1.0)+ base_param.data / (self.n_models + 1.0)
                
                sq_mean = sq_mean * self.n_models / (
                    self.n_models + 1.0) + base_param.data ** 2 / (self.n_models + 1.0)

                module.__setattr__("%s_mean" % name, mean)
                module.__setattr__("%s_sq_mean" % name, sq_mean)
                new_params = (base_param.data - mean).view(-1, 1).T
                deviation_matrix = module.__getattr__("%s_deviation_matrix" % name)

                if(deviation_matrix.size(dim = 0) == self.hyperparameters.k):
                    module.__setattr__("%s_deviation_matrix" % name, torch.cat(
                        (deviation_matrix[:self.hyperparameters.k - 1,:], new_params), axis = 0))
                else:
                    module.__setattr__("%s_deviation_matrix" % name, torch.cat(
                        (deviation_matrix, new_params), axis = 0))



        self.n_models += 1
