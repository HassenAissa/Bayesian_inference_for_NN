import torch
import torch.nn
import sys
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src/optimizer')

from SWAG import SWAG, SwagHyperparam
import numpy as np


class Data:
    def __init__(self, input, label):
        self.input = torch.cat(input)
        self.label = torch.cat(label)

base_model = torch.nn.Linear(20, 2, bias=True)
expected = torch.randn(1, 2)
training_size = 10000
training_input = []
training_label = []

for i in range(training_size):
    training_input.append(torch.randn(1, 20))
    training_label.append(expected)



hyperparameters = SwagHyperparam(
    k = 10,
    frequency = 1,
    scale=1,
    lr = 1e-2
)

data = Data(
    training_input,
    training_label
)


swag_model = SWAG(
    base_model = base_model,
    data = data,
    hyperparameters  = hyperparameters,
)

input = torch.randn(1, 20)


for _ in range(10000):
    swag_model.step()


print("finished training")

sum = 0
nb_samples = 20000

for i in range(nb_samples):
    distribution = swag_model.distribution()
    out = swag_model.predict(distribution, input)
    sum += out

loss = ((expected - out) ** 2.0).sum()
print("loss ",loss)
print("expected ", expected)
print("prediction " , sum/nb_samples)

if(loss < 0.01):
    print("test successful")
else:
    print("test FAILED !!!!!!!")
