import tensorflow as tf

from Pyesian.distributions import Distribution


class Constant(Distribution):
    def __init__(self, value: tf.Tensor, size: int):
        self._value = value
        super().__init__(int(tf.math.reduce_sum(value).numpy()))

    def sample(self) -> tf.Tensor:
        return self._value

    def serialize(self) -> str:
        pass

    @classmethod
    def deserialize(cls, data: str) -> 'Distribution':
        pass