import json

import tensorflow as tf
import tensorflow_probability as tfp

from src.distributions.Distribution import Distribution


class TensorflowProbabilityDistribution(Distribution):

    def __init__(self, tf_distribution: tfp.distributions.Distribution):
        self._tf_distribution = tf_distribution
        size = 0
        if len(tf_distribution.batch_shape) + len(tf_distribution.event_shape) != 1:
            raise ValueError('The provided tensorflow distribution should be a vector')
        if len(tf_distribution.event_shape) >= 1:
            size = tf_distribution.event_shape[0]
        else:
            size = tf_distribution.batch_shape[0]
        super().__init__(size)


    @classmethod
    def __DISTRIBUTION_SERIALIZER_REGISTER(cls):
        return {

        }

    @classmethod
    def __DEFAULT_BASE_SERIALIZER(cls):
        # python logic...
        from src.distributions.tf.BaseSerializer import BaseSerializer
        return BaseSerializer()

    def serialize(self) -> str:
        distribution_type = type(self._tf_distribution).__name__
        if distribution_type in self.__DISTRIBUTION_SERIALIZER_REGISTER().keys():
            return self.__DISTRIBUTION_SERIALIZER_REGISTER()[distribution_type].serialize(self._tf_distribution)
        else:
            return self.__DEFAULT_BASE_SERIALIZER().serialize(self._tf_distribution)

    @classmethod
    def deserialize(cls, data: str) -> 'Distribution':
        dtbn_dict = json.loads(data)
        if dtbn_dict["type"] in cls.__DISTRIBUTION_SERIALIZER_REGISTER():
            return cls.__DISTRIBUTION_SERIALIZER_REGISTER()[dtbn_dict["type"]].deserialize(data)
        else:
            return cls.__DEFAULT_BASE_SERIALIZER().deserialize(data)

    def sample(self) -> tf.Tensor:
        vector = self._tf_distribution.sample()
        tf.debugging.check_numerics(vector, "distribution failed")
        return self._tf_distribution.sample()

