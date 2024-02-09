import tensorflow_probability as tfp
import tensorflow as tf
import json

from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution


class BaseSerializer:
    """
    a class providing a basic unoptimised serializer for any tensorflow_probability distribution
    """
    def __init__(self):
        pass

    def _serialise_parameter(self, value):
        if isinstance(value, tf.Tensor):
            return value.numpy().tolist()
        return value

    def serialize(self, tf_distribution: tfp.distributions.Distribution) -> str:
        """
        serialize a tfp distribution

        Args:
            tf_distribution (tfp.distributions.Distribution): the distribution to be serialized

        Returns:
            str: the serialized distribution
        """
        dtbn_dict = {
            'type': type(tf_distribution).__name__,
            'params': {k: self._serialise_parameter(v) for k, v in tf_distribution.parameters.items()}
        }
        return json.dumps(dtbn_dict)

    def deserialize(self, cls: str) -> TensorflowProbabilityDistribution:
        """
        deserialize a tfp distribution

        Args:
            cls (str): the seriliazed distribution

        Returns:
            TensorflowProbabilityDistribution: the deserialized distribution
        """
        dtbn_dict = json.loads(cls)
        tf_distribution = getattr(tfp.distributions, dtbn_dict['type'])(**dtbn_dict['params'])
        return TensorflowProbabilityDistribution(tf_distribution)
