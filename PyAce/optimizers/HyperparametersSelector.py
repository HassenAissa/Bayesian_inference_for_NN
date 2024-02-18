import copy
from functools import cmp_to_key
import math
from PyAce.datasets import Dataset
from . import Optimizer
from . import HyperParameters
import numpy as np
import matplotlib.pyplot as plt


class HyperParametersSelector:
    def __init__(self, dataset: Dataset,nb_samples = 6, **kwargs):
        self._nb_samples = nb_samples
        self._params = copy.deepcopy(kwargs)
        self._dataset = dataset
        self._hyperparameters_list = []

    def __getattr__(self, item):
        if item in self._params:
            return self._params[item]
        else:
            raise AttributeError("'HyperParameters' object has no attribute " + str(item))
    
    def _recursive_cross_validation(self, arguments, i):
        if len(arguments) == len(self._params)-1:
            arg = "lr"
            start = self._params[arg][0]
            end = self._params[arg][1]
            for val in sorted(np.random.uniform(start, end, [self._nb_samples,])):
                hyperparameters = HyperParameters(**{**arguments, **{arg : val}})
                self._hyperparameters_list.append(hyperparameters)
        else:    
            arg = list(self._params.keys())[i]
            if arg != "lr":
                start = self._params[arg][0]
                end = self._params[arg][1]
                for val in sorted(np.random.uniform(start, end, [self._nb_samples,])):
                    self._recursive_cross_validation({**arguments, **{arg : val}}, i+1) 
            else:
                self._recursive_cross_validation(arguments, i+1)

    def _compare(self,x, y):
        if math.isclose(x[0], y[0]) == False:
            return x[0] - y[0]   
        else:
            return 1

    def cross_validation(self, model_config, optimizer_class: Optimizer, **kwargs):
        for i in range(self._nb_samples):
            self._recursive_cross_validation({}, 0)  
        
        results = []
        for hyperparameters in self._hyperparameters_list:
            optimizer = optimizer_class()
            optimizer.compile(hyperparameters, model_config, self._dataset, **kwargs)
            optimizer.train(10)
            bayesian_model = optimizer.result()
            x, y_true = next(iter(self._dataset.valid_data.batch(self._dataset.valid_data.cardinality())))
            _,prediction = bayesian_model.predict(x, 100)
            results.append((hyperparameters._params["lr"], self._dataset.loss()(y_true, prediction), hyperparameters._params))
        results = sorted(results, key = cmp_to_key(self._compare))
        nb_elems = len(results)//self._nb_samples
        losses_splitted = []
        parameters = []
        lr_splitted = []      
        for i in range(self._nb_samples):
            current = i
            lrs = []
            losses = []
            parameters.append({key:results[i][2][key] for key in results[i][2] if key!='lr'})
            while(current < len(results)):
                lrs.append(results[current][0])
                losses.append(results[current][1])
                current += nb_elems
            losses_splitted.append(losses)
            lr_splitted.append(lrs)
        for lrs, params, loss in zip(lr_splitted, parameters, losses_splitted):
            plt.plot(lrs, loss, label=str(params))
        plt.xlabel("lr")
        plt.ylabel("loss")
        plt.legend()
        plt.show()            
        print(results)