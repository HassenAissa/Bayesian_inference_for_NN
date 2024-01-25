from abc import ABC, abstractmethod
import tensorflow as tf

from src.distributions.tf import Sampled
from src.distributions.tf.Constant import Constant
from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution


class Distribution(ABC):
    __DISTRIBUTION_REGISTER: dict[str, 'Distribution'] = {
        "TensorflowProbabilityDistribution" : TensorflowProbabilityDistribution,
        "Constant" : Constant,
        "Sampled" : Sampled
    }
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

    @classmethod
    def deserialize_from(cls, name: str, extension_register: dict,data: str) -> 'Distribution':
        if name in cls.__DISTRIBUTION_REGISTER:
            return cls.__DISTRIBUTION_REGISTER[name].deserialize(data)
        else:
            return extension_register[name].deserialize(data)



