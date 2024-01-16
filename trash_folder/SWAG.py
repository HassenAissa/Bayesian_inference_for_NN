import torch
import torch.nn as nn
import numpy as np
import Bayesian_NN.git_version.Beyesian_inference_for_NN.trash_folder.utils as utils


def swag_parameters(module, params):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_deviation_matrix" % name, data.new_empty((0, data.numel())).zero_())

        params.append((module, name))
    

class SWAG(nn.Module):
    def __init__(self, base,base_model,input, expected, k =10, *args, **kwargs):
        super(SWAG, self).__init__()

        self.params = list()
        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.base = base(*args, **kwargs)
        self.k = k
        self.base.apply(lambda module: swag_parameters(module=module, params=self.params))
        self.base_model = base_model
        self.optimizer = torch.optim.SGD(self.base_model.parameters(), lr=1e-3)
        self.expected = expected
        self.input = input
        #print(self.base_model(input))





    def forward(self,*args, **kwargs):
        return self.base(*args, **kwargs)

    

    def sample (self):
        mean_list = []
        sq_mean_list = []
        deviation_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)
            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())
            deviation = module.__getattr__("%s_deviation_matrix" % name)
            deviation_list.append(deviation.cpu())

        mean = utils.flatten(mean_list)
        sq_mean = utils.flatten(sq_mean_list)
        deviation = torch.cat(deviation_list, dim=1)
        deviation_matrix = deviation.T.matmul(deviation)
        scale = 0.1
        distribution = torch.distributions.MultivariateNormal(mean, 0.5*torch.diag(sq_mean-mean**2.0) +scale* 0.5*deviation_matrix/(self.k-1))
        sample = distribution.sample()
        sample = sample.unsqueeze(0)
        samples_list = utils.unflatten_like(sample, mean_list)
        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample)
        
            


    def step(self):
        input = torch.randn(1, 20)
        self.base_model.zero_grad()
        output = self.base_model(input)
        loss = ((self.expected - output) ** 2.0).sum()
        print(loss)
        loss.backward()
        self.optimizer.step()
        for (module, name), base_param in zip(self.params, self.base_model.parameters()):
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            mean = mean * self.n_models.item() / (self.n_models.item() + 1.0) + base_param.data / (self.n_models.item() + 1.0)
            sq_mean = sq_mean * self.n_models.item() / (self.n_models.item() + 1.0) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
            new_params = (base_param.data - mean).view(-1, 1).T
            deviation_matrix = module.__getattr__("%s_deviation_matrix" % name)

            if(deviation_matrix.size(dim = 0) % self.k == 0):
                module.__setattr__("%s_deviation_matrix" % name, torch.cat((deviation_matrix[:self.k - 1,:], new_params), axis = 0))
            else:
                module.__setattr__("%s_deviation_matrix" % name, torch.cat((deviation_matrix, new_params), axis = 0))



        self.n_models.add_(1)
