from math import sqrt
import json
import tensorflow as tf
import tensorflow_probability as tfp

from src.distributions.Distribution import Distribution


class MultivariateNormalDiagPlusLowRank(Distribution):
    def serialize(self) -> str:
        to_json = {"mean": self._mean.numpy().tolist(), "D": self._D.numpy().tolist(), "diag": self._diag.numpy().tolist()}
        return json.dumps(to_json)
        
    @classmethod
    def deserialize(cls, data: str) -> 'Distribution':
        from_json = json.loads(data)
        return MultivariateNormalDiagPlusLowRank(tf.convert_to_tensor(from_json["mean"]), tf.convert_to_tensor(from_json["diag"]),
                                                  tf.convert_to_tensor(from_json["D"]))

    def __init__(self, mean: tf.Tensor, diag: tf.Tensor, D: tf.Tensor):
        super().__init__(mean.shape[0])
        self._mean = mean
        self._D = D
        self._diag = diag

    def sample(self) -> tf.Tensor:
        k = self._D.shape[1]
        z1 = tfp.distributions.Normal(tf.zeros_like(self._mean), self._diag).sample()
        z2 = tfp.distributions.Normal(tf.zeros((k,)), tf.ones(k, )).sample()
        z1 = tf.reshape(z1, (-1,1))
        z2 = tf.reshape(z2, (-1,1))
        cov_mean = (tf.linalg.matmul(self._D, z2)) * sqrt(1 / (2 * (k - 1)))

        res = tf.reshape(self._mean, (-1,1)) + z1 + cov_mean
        return tf.reshape( res, (-1,))
