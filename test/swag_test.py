import torch
import torch.nn
import sys
import pytest

sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src/optimizer')

from SWAG import SWAG, SwagHyperparam
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def test_swag_on_distribution_succeed():


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
        lr = 1e-2,
        loss = torch.nn.MSELoss()
    )


    train_dataset = TensorDataset(torch.cat(training_input), torch.cat(training_label))
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)



    swag_model = SWAG(
        base_model = base_model,
        dataloader = dataloader,
        hyperparameters  = hyperparameters,
    )

    input = torch.randn(1, 20)


    for _ in range(20000):
        swag_model.step()


    print("finished training")

    sum = 0
    nb_samples = 30000

    for i in range(nb_samples):
        mean, var, deviation = swag_model.distribution()
        out = swag_model.predict(mean, var, deviation, input)
        sum += out

    loss = ((expected - out) ** 2.0).sum()
    print("loss ",loss)
    print("expected ", expected)
    print("prediction " , sum/nb_samples)

    assert loss < 0.05
    if(loss < 0.05):
        print("test successful")
    else:
        print("test FAILED !!!!!!!")


test_swag_on_distribution_succeed()