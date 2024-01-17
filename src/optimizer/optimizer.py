from abc import ABC, abstractmethod

class Optimizer(ABC): #may inherit 
    @abstractmethod
    def __init__(self, hyperparameters):
        pass
    @abstractmethod
    def hyperparam_init():
        pass
    @abstractmethod
    def step():
        pass
    @abstractmethod
    def update_hyperparam():
        pass