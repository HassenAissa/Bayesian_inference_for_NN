from abc import ABC
import tensorflow as tf
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, optimizer : tf.keras.optimizers):
        self._optimizer = optimizer

    @abstractmethod
    def get_trainable_variables(self) -> list[tf.Tensor]:
        pass

    @abstractmethod
    def optimize(self, grad: list[tf.Tensor], trainable_variables: list[tf.Tensor]):
        pass
    
    @abstractmethod
    def predict(self, feature, training=False):
        pass



