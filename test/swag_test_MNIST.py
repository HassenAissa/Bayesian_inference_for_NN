import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src')
sys.path.append('C:/Users/hasse/Documents/Hassen/SEGP/Bayesian_NN/git_version/Beyesian_inference_for_NN/src/optimizer')

from SWAG import SWAG, SwagHyperparam
import utils
import numpy as np
from torch.utils.data import TensorDataset, DataLoader



class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, filters=(16, 32, 64), kernel_size=3, features=120, n_conv = 3, n_linear = 2):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        padding_size =  int((kernel_size - 1) / 2)


        #3Conv & 2Linear
        self.conv2d1 = nn.Conv2d(input_channels, filters[0], kernel_size, padding=padding_size)
        self.conv2d2 = nn.Conv2d(filters[0], filters[1], kernel_size, padding=padding_size)
        self.conv2d3 = nn.Conv2d(filters[1], filters[2], kernel_size, padding=padding_size)
        
        f_size = int(32 / (2**n_conv))
        self.fc1 = nn.Linear(filters[2]*f_size*f_size, features)
        self.fc2 = nn.Linear(features, n_classes)
        
        #2Conv & 3Linear
        # self.conv2d1 = nn.Conv2d(input_channels, filters[0], kernel_size, padding=padding_size)
        # self.conv2d2 = nn.Conv2d(filters[0], filters[1], kernel_size, padding=padding_size)
                
        # f_size = int(32 / (2**2))
        # self.fc1 = nn.Linear(filters[1]*f_size*f_size, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, n_classes)
        
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """

        #3Conv & 2Linear
        
        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2d3(x)), 2)
        x = x.flatten(-3)
        x = F.relu(self.fc1(x))
        preds = self.fc2(x)
        return preds

        #2Conv & 3Linear
        # x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        # x = x.flatten(-3)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # preds = self.fc3(x)
        # return preds



def test_swag_on_MNIST():
    xtrain, xtest, ytrain, ytest = utils.load_data(r"Bayesian_NN\git_version\Beyesian_inference_for_NN\test\dataset_HASYv2")
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    
    # normalize, add bias
    means = np.mean(xtrain, axis = 0, keepdims= True)
    stds = np.std(xtrain, axis = 0, keepdims= True)
    
    xtrain = utils.normalize_fn(xtrain, means, stds)
    xtest = utils.normalize_fn(xtest, means, stds)
    fraction_train = 0.8
    n_samples = xtrain.shape[0]
    rinds = np.random.permutation(n_samples)

    n_train = int(n_samples * fraction_train)
    xtest = xtrain[rinds[n_train:]]
    ytest = ytrain[rinds[n_train:]]

    xtrain = xtrain[rinds[:n_train]]
    ytrain = ytrain[rinds[:n_train]]
    n_classes = utils.get_n_classes(ytrain)
    xtrain = xtrain.reshape(xtrain.shape[0], 1, 32, 32)
    xtest = xtest.reshape(xtest.shape[0], 1, 32, 32)

    base_model = CNN(xtrain.shape[1], n_classes)

    hyperparameters = SwagHyperparam(
        k = 10,
        frequency = 1,
        scale=1,
        lr = 1e-2,
        loss = torch.nn.CrossEntropyLoss()
    )


    train_dataset = TensorDataset(torch.from_numpy(xtrain).float(), torch.from_numpy(ytrain).long())
    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    swag_model = SWAG(
        base_model = base_model,
        dataloader = dataloader,
        hyperparameters  = hyperparameters,
    )


    for _ in range(5000):
        swag_model.step()

    print("finished training")
    nb_samples = 5000

    test_dataset = TensorDataset(torch.from_numpy(xtest).float())
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    prediction = torch.zeros_like(torch.from_numpy(xtest))
    
    prediction = torch.zeros([xtest.shape[0], n_classes])
    for i in range(nb_samples):
        mean, var, deviation = swag_model.distribution()
        # print("found distribution")
        with torch.no_grad():
            pred_labels = []
            # for _,b in enumerate(test_dataloader):
            samples = torch.from_numpy(xtest).float()
            pred = swag_model.predict(mean, var, deviation, samples)
            prediction += pred

    prediction /= nb_samples
    print(torch.argmax(prediction, dim = -1))
    print(torch.sum(torch.argmax(prediction, dim = -1) == torch.from_numpy(ytest))/xtest.shape[0])
test_swag_on_MNIST()
