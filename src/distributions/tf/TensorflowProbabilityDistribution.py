import json

import tensorflow as tf
import tensorflow_probability as tfp

from src.distributions.Distribution import Distribution
from src.distributions.tf.BaseSerializer import BaseSerializer

DISTRIBUTION_SERIALIZER_REGISTER: [str, BaseSerializer] = {

}
DEFAULT_BASE_SERIALIZER = BaseSerializer()


class TensorflowProbabilityDistribution(Distribution):
    def __init__(self, tf_distribution: tfp.distributions.Distribution):
        self._tf_distribution = tf_distribution
        size = 0
        if len(tf_distribution.batch_shape) + len(tf_distribution.event_shape) != 1:
            raise ValueError('The provided tensorflow distribution should be a vector')
        if len(tf_distribution.event_shape) > 1:
            size = tf_distribution.event_shape[0]
        else:
            size = tf_distribution.batch_shape[0]
        super().__init__(size)

    def serialize(self) -> str:
        distribution_type = type(self._tf_distribution).__name__
        if distribution_type in DISTRIBUTION_SERIALIZER_REGISTER.keys():
            return DISTRIBUTION_SERIALIZER_REGISTER[distribution_type].serialize(self._tf_distribution)
        else:
            return DEFAULT_BASE_SERIALIZER.serialize(self._tf_distribution)

    @classmethod
    def deserialize(cls, data: str) -> 'Distribution':
        dist_dict = json.loads(data)
        if dist_dict["type"] in DISTRIBUTION_SERIALIZER_REGISTER:
            return DISTRIBUTION_SERIALIZER_REGISTER[dist_dict["type"]].deserialize(data)
        else:
            return DEFAULT_BASE_SERIALIZER.deserialize(data)

    def sample(self) -> tf.Tensor:
        pass
