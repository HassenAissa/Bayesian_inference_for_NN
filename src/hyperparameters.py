from abc import ABC, abstractmethod


class Hyperparams(ABC):
    @abstractmethod
    def __init__(self):
        pass
