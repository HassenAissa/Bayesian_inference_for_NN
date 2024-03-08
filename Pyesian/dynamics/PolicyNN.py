import tensorflow as tf

from Pyesian.dynamics.Policy import Policy
class PolicyNN(Policy):
    def __init__(self, model, optimizer):
        self._model = model
    def get_trainable_variables(self) -> list[tf.Tensor]:
        return list(self._model.trainables_variables)

    def optimize(self, gradients : list[tf.Tensor]):
        self._optimizer.appy_gradients(zip(gradients, self._model.trainable_variables))

    def predict(self, feature, training=False):
        probability = self._model(feature, training = training)
        return tf.argmax(probability), probability
