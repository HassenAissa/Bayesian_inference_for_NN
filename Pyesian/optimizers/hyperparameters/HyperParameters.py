import copy
# import tensorflow as tf
import numpy as np


class HyperParameters:
    """
        This class represents the hyperparameters for an Optimizer
        Args:
            batch_size: the batch size. Defaults to 64.
            **kwargs: the hyperparameters of the Optimizer
    """

    def __init__(self, **kwargs):    
        self._params = copy.deepcopy(kwargs)
        if "batch_size" not in kwargs:
            self._params["batch_size"] = 64
        self.connectors = "._-"

    def __getattr__(self, item):
        if item in self._params:
            return self._params[item]
        else:
            raise AttributeError("'HyperParameters' object has no attribute " + str(item))
    
    def from_file(self, fn):
        f = open(fn, "r")
        text = f.read()
        f.close()
        return self.parse(text)

    def parse(self, text:str):
        keys = []
        values = []
        k = ""
        v = ""
        s = 0
        for c in text:
            if s == 0:
                if c.isalnum() or c in self.connectors:
                    k += c
                elif k:
                    keys.append(k)
                    k = ""
                    s = 1
            else:
                if c.isdigit() or c in "-.":
                    v += c
                elif v:
                    values.append(float(v))
                    v = ""
                    s = 0
        if k:
            keys.append(k)
            for i in range(len(keys) - len(values)):
                values.append(0.0)
        elif v:
            values.append(float(v))
        
        for i in range(len(keys)):
            self._params[keys[i]] = values[i]
        return self
    
# res = HyperParameters().from_file("static/hyperparams/swag.txt")
# print(res._params)



                



        
        









