import tensorflow_probability as tfp
import json

from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution


class BaseSerializer:
    def __init__(self):
        pass

    def serialize(self, tf_distribution: tfp.distributions.Distribution) -> str:
        dist_dict = {
            'type': type(tf_distribution).__name__,
            'params': {k: v.numpy().tolist() for k, v in tf_distribution.parameters.items()}
        }
        return json.dumps(dist_dict)

    def deserialize(self, cls: str) -> TensorflowProbabilityDistribution:
        dist_dict = json.loads(cls)
        tf_distribution = tfp.distributions.getattr(dist_dict['type'])(**dist_dict['params'])
        return TensorflowProbabilityDistribution(tf_distribution)
