from abc import ABC, abstractmethod
import tensorflow as tf



class Distribution(ABC):
    def __init__(self, size: int):
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    @abstractmethod
    def sample(self) -> tf.Tensor:
        pass

    @abstractmethod
    def serialize(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def deserialize(cls, data: str) -> 'Distribution':
        pass





