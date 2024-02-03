from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self) -> None:
        super().__init__()