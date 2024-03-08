from .Distribution import Distribution
from .Sampled import Sampled
from .tf.Constant import Constant
from .MultivariateNormalDiagPlusLowRank import MultivariateNormalDiagPlusLowRank

from .tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution


class DistributionSerializer:
    """A class allowing to switch between distribution serializers
    """
    __DISTRIBUTION_REGISTER: dict[str, 'Distribution'] = {
        "MultivariateNormalDiagPlusLowRank": MultivariateNormalDiagPlusLowRank,
        "TensorflowProbabilityDistribution": TensorflowProbabilityDistribution,
        "Constant": Constant,
        "Sampled": Sampled
    }

    @classmethod
    def load_from(cls, name: str, extension_register: dict, path: str) -> 'Distribution':
        """selects the class from which to load the distribution by using the predefined distribution register \
        or the user's extesion_registers

        Args:
            name (str): the name of the class of the distribution to load
            extension_register (dict): the user defned mapping between a distribution name and a distribution class
            path (str): the path from which to load the distribution

        Returns:
            Distribution: the loaded distribution
        """
        if name in cls.__DISTRIBUTION_REGISTER:
            return cls.__DISTRIBUTION_REGISTER[name].load(path)
        else:
            return extension_register[name].load(path)
