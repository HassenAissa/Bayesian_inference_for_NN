import copy
# import tensorflow as tf
import numpy as np


class HyperParameters:
    """
    Kind of a useless class for the moment as it just represents a dict.
    An instance of parameters needs to be passed to the optimizer

    This class will be used in the future to improve the parameters and do some parameters optimization
    and analysis on the parameters. It might even allow to do AutoML in the future.

    THIS CLASS IS IMMUTABLE !!!! TRUST ME!
    """

    def __init__(self, **kwargs):
        self._params = copy.deepcopy(kwargs)
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



                



        
        









