import tensorflow_probability as tfp
import json

from src.distributions.tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution


class BaseSerializer:
    def __init__(self):
        pass

    def serialize(self, tf_distribution: tfp.distributions.Distribution) -> str:
        dtbn_dict = {
            'type': type(tf_distribution).__name__,
            'params': {k: v.numpy().tolist() for k, v in tf_distribution.parameters.items()}
        }
        return json.dumps(dtbn_dict)

    def deserialize(self, cls: str) -> TensorflowProbabilityDistribution:
        dtbn_dict = json.loads(cls)
        tf_distribution = tfp.distributions.getattr(dtbn_dict['type'])(**dtbn_dict['params'])
        return TensorflowProbabilityDistribution(tf_distribution)
