from .Distribution import Distribution
from .Sampled import Sampled
from .tf.Constant import Constant
from .MultivariateNormalDiagPlusLowRank import MultivariateNormalDiagPlusLowRank

from .tf.TensorflowProbabilityDistribution import TensorflowProbabilityDistribution


class DistributionSerializer:
    __DISTRIBUTION_REGISTER: dict[str, 'Distribution'] = {
        "MultivariateNormalDiagPlusLowRank": MultivariateNormalDiagPlusLowRank,
        "TensorflowProbabilityDistribution": TensorflowProbabilityDistribution,
        "Constant": Constant,
        "Sampled": Sampled
    }

    @classmethod
    def load_from(cls, name: str, extension_register: dict, path: str) -> 'Distribution':
        if name in cls.__DISTRIBUTION_REGISTER:
            return cls.__DISTRIBUTION_REGISTER[name].load(path)
        else:
            return extension_register[name].load(path)
