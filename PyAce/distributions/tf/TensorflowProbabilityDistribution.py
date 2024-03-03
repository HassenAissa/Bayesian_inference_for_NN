import json

import tensorflow as tf
import tensorflow_probability as tfp
import os
from PyAce.distributions import Distribution


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
        from PyAce.distributions.tf.BaseSerializer import BaseSerializer
        return BaseSerializer()

    def store(self, path: str) -> str:
        distribution_type = type(self._tf_distribution).__name__
        data = ""
        if distribution_type in self.__DISTRIBUTION_SERIALIZER_REGISTER().keys():
            data = self.__DISTRIBUTION_SERIALIZER_REGISTER()[distribution_type].serialize(self._tf_distribution)
        else:
            data = self.__DEFAULT_BASE_SERIALIZER().serialize(self._tf_distribution)
        with open(os.path.join(path, "distribution.json"), "w") as file:
            file.write(data)

    @classmethod
    def load(cls, path: str) -> 'Distribution':
        data = ""
        with open(os.path.join(path, "distribution.json"), "r") as file:
            data = file.read()
        dtbn_dict = json.loads(data)
        if dtbn_dict["type"] in cls.__DISTRIBUTION_SERIALIZER_REGISTER():
            return cls.__DISTRIBUTION_SERIALIZER_REGISTER()[dtbn_dict["type"]].deserialize(data)
        else:
            return cls.__DEFAULT_BASE_SERIALIZER().deserialize(data)

    def sample(self) -> tf.Tensor:
        vector = self._tf_distribution.sample()
        # tf.debugging.check_numerics(vector, "distribution failed")
        return self._tf_distribution.sample()

